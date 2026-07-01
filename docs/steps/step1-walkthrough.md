# Walkthrough do Step 1 — Transcrição + Diarização

## O que este passo faz

O Step 1 pega o ficheiro de vídeo (`input/meeting.mp4`) e produz um JSON
(`output/transcript.json`) com tudo o que foi dito na reunião, quem disse o quê,
e em que instantes. Para isso, encadeia três operações: (1) extrair o áudio do
vídeo com `ffmpeg`, (2) transcrever esse áudio para texto com o
WhisperX (um modelo de IA que percebe voz humana), e (3) identificar quem fala
em cada momento com o pyannote (diarização de oradores). No final, cruza os
resultados da transcrição com os da diarização para associar cada frase a uma
pessoa.

O pipeline tem 4 ficheiros envolvidos. Vamos vê-los um a um.

---

## 1. `pipeline/audio.py` — Extrair áudio do vídeo

Serviço simples: chama o programa `ffmpeg` para converter o vídeo num ficheiro
WAV que os modelos de IA conseguem processar.

### Imports

```python
import subprocess
import os
```

- `subprocess` — permite correr programas externos (neste caso o `ffmpeg`)
  a partir do Python.
- `os` — funções para interagir com o sistema operativo, como criar pastas
  ou verificar se ficheiros existem.

### Constantes

```python
AUDIO_SAMPLE_RATE = 16000
AUDIO_TEMP_FILE = "output/audio_temp.wav"
```

- `AUDIO_SAMPLE_RATE = 16000` — 16 kHz é a frequência de amostragem padrão
  para modelos de voz (16000 amostras por segundo). É o suficiente para
  captar a fala humana e evita ficheiros enormes.
- `AUDIO_TEMP_FILE` — caminho onde o ficheiro WAV temporário vai ser guardado.

### Função `extract_audio()`

```python
def extract_audio(video_path: str, output_path: str = None) -> str:
```

Recebe o caminho do vídeo e um caminho de saída opcional. Devolve o caminho
do ficheiro de áudio criado.

```python
    if output_path is None:
        output_path = AUDIO_TEMP_FILE
```

Se não foi dado um caminho específico, usa o `AUDIO_TEMP_FILE` definido acima.

```python
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

Garante que a pasta `output/` existe. `exist_ok=True` significa que não dá erro
se ela já existir.

```python
    cmd = [
        "ffmpeg",
        "-y",                      # sobrescreve ficheiros existentes sem perguntar
        "-i", video_path,          # ficheiro de entrada (o vídeo)
        "-ar", str(AUDIO_SAMPLE_RATE),  # converte para 16 kHz
        "-ac", "1",                # um canal áudio (mono) — só precisamos de voz
        "-sample_fmt", "s16",      # formato de amostragem: signed 16-bit
        output_path,               # ficheiro de saída
    ]
```

Constrói a lista de argumentos para o `ffmpeg`. Cada argumento é um elemento
da lista. `-y` diz ao ffmpeg para não perguntar antes de substituir ficheiros.

```python
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install with: sudo apt-get install ffmpeg"
        )
```

`subprocess.run()` executa o comando `ffmpeg` e espera que ele termine.
`capture_output=True` captura o que o ffmpeg escreve no ecrã.
Se o `ffmpeg` não estiver instalado no sistema, o Python lança
`FileNotFoundError` — nós transformamos isso numa mensagem clara.

```python
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
```

Se o ffmpeg correu mas deu erro (returncode diferente de 0), lançamos uma
exceção com a mensagem de erro.

```python
    return output_path
```

Devolve o caminho do ficheiro WAV criado para quem chamou a função poder
usá-lo.

---

## 2. `pipeline/diarize.py` — Identificar quem fala quando

A diarização responde a "quem falou e durante quanto tempo?".

### Imports

```python
import os
import torch
```

- `os` — verificar se o ficheiro de áudio existe.
- `torch` — PyTorch, a biblioteca de machine learning que o pyannote usa.

### Constante

```python
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
```

Nome do modelo pré-treinado no HuggingFace que vamos usar. A versão 3.1 é a
mais recente do pyannote.

### Função `diarize_audio()`

```python
def diarize_audio(audio_path: str, hf_token: str) -> list[dict]:
```

Recebe o caminho do áudio e um token de autenticação do HuggingFace. O token é
necessário porque o modelo pyannote exige que o utilizador aceite os termos de
uso no site do HuggingFace.

Devolve uma lista de dicionários, cada um com `{speaker, start, end}`.

```python
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not hf_token:
        raise ValueError("HF_TOKEN is required for diarization. Set it in .env")
```

Validações básicas: se o áudio não existe ou se não há token, para já com erro.

```python
    # Deferred import: pyannote.audio loads torch at import time and crashes
    # in CPU-only environments without CUDA libs.
    from pyannote.audio import Pipeline
```

**Importante:** O import do pyannote está dentro da função, e não no topo do
ficheiro. Isto é propositado — o pyannote carrega o PyTorch logo ao ser
importado, e isso pode crashar em ambientes sem GPU (sem CUDA). Ao adiar o
import para dentro da função, garantimos que validamos os argumentos primeiro
e que a CLI faz `sys.exit` com uma mensagem clara antes de o crash acontecer.

```python
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
```

Carrega o modelo de diarização pré-treinado do HuggingFace. O `hf_token` é
passado para autenticar o download.

```python
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
```

Se houver GPU disponível, move o modelo para a GPU para ser mais rápido.
Caso contrário, fica na CPU (mais lento, mas funciona).

```python
    diarization = pipeline(audio_path)
```

Executa a diarização: o modelo analisa o áudio e descobre quem fala em cada
intervalo de tempo. O resultado é um objeto especial do pyannote.

```python
    return [
        {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
    ]
```

Converte o resultado do pyannote para uma lista simples de dicionários Python.
`itertracks(yield_label=True)` percorre todos os segmentos de voz. Para cada
segmento:
- `turn` — contém `start` e `end` (início e fim do segmento em segundos).
- `speaker` — rótulo do orador (ex: `SPEAKER_00`, `SPEaker_01`).
- `round(..., 3)` — limita o número de casas decimais a 3.

O resultado é algo como:
```python
[
    {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2},
    {"speaker": "SPEAKER_01", "start": 3.5, "end": 8.1},
    ...
]
```

---

## 3. `pipeline/transcribe.py` — Transcrição + junção com oradores

Ficheiro com duas funções: transcrever o áudio para texto e depois juntar
esses textos com os oradores da diarização.

### Imports

```python
import os
import torch
import whisperx
```

- `whisperx` — biblioteca que estende o modelo Whisper da OpenAI com
  alinhamento ao nível da palavra (sabemos quando cada palavra começa e acaba).

### Constantes

```python
WHISPER_MODEL = "large-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
```

- `WHISPER_MODEL = "large-v2"` — versão do modelo Whisper. "large-v2" é o
  maior e mais preciso (mas também o mais lento).
- `DEVICE` — deteta automaticamente se há GPU. Se sim, usa `"cuda"`, senão usa
  `"cpu"`.
- `COMPUTE_TYPE` — define a precisão dos cálculos: `"float16"` (16-bit) na GPU,
  `"int8"` (8-bit) na CPU. Isto acelera o processamento sem perder muita
  qualidade.

### Função `transcribe_audio()`

```python
def transcribe_audio(audio_path: str) -> list[dict]:
```

Recebe o caminho do áudio e devolve uma lista de segmentos com texto,
timestamps e palavras individuais. Cada segmento corresponde a uma frase ou
parte de frase.

```python
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
```

Se o ficheiro de áudio não existe, dá erro.

```python
    model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio_path)
```

Carrega o modelo WhisperX e transcreve o áudio inteiro. O `result` contém
tudo o que foi dito, segmentado por pausas naturais, incluindo o texto e os
timestamps aproximados.

```python
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=DEVICE
    )
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device=DEVICE)
```

O WhisperX dá timestamps ao nível do segmento, mas nós queremos timestamps ao
nível da palavra (para depois cruzar com a diarização). Para isso, carregamos
um modelo de alinhamento (`align_model`) específico para o idioma detetado
(`result["language"]`) e depois corremos o alinhamento. O `aligned["segments"]`
contém os segmentos originais mas cada um com uma lista `words` onde cada
palavra tem `{word, start, end}`.

```python
    return aligned["segments"]
```

Devolve os segmentos alinhados. Cada segmento tem este formato:
```python
{
    "start": 0.5,
    "end": 3.2,
    "text": "Bom dia a todos",
    "words": [
        {"word": "Bom", "start": 0.5, "end": 0.8},
        {"word": "dia", "start": 0.9, "end": 1.2},
        {"word": "a", "start": 1.3, "end": 1.4},
        {"word": "todos", "start": 1.5, "end": 3.2}
    ]
}
```

Nota: não há campo `speaker` aqui — isso é tratado na função seguinte.

### Função `merge_speaker_labels()`

```python
def merge_speaker_labels(segments: list[dict], diarization: list[dict]) -> list[dict]:
```

Junta os segmentos da transcrição com os oradores da diarização.

Recebe:
- `segments` — lista de segmentos de texto com timestamps (vindos do WhisperX).
- `diarization` — lista de segmentos de orador (vindos do pyannote).

Devolve uma nova lista de segmentos com o campo `speaker` adicionado.

```python
    result = []
    for seg in segments:
```

Para cada segmento de texto...

```python
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
```

...começamos por assumir que o orador é desconhecido. Vamos procurar qual o
orador que mais se sobrepõe temporalmente com este segmento.

```python
        for d in diarization:
            overlap = min(seg["end"], d["end"]) - max(seg["start"], d["start"])
```

Para cada segmento de diarização, calculamos o tempo de sobreposição: o fim
do mais cedo (min) menos o início do mais tarde (max). Se não houver
sobreposição, o resultado é negativo.

```python
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]
```

Se esta sobreposição for maior do que a melhor que já encontrámos (e positiva),
guardamos o orador correspondente.

```python
        result.append({**seg, "speaker": best_speaker})
```

Criamos um novo dicionário com todos os campos do segmento original **mais**
o campo `speaker`. Usamos `{**seg, ...}` que é o "operador de espalhamento"
— copia todas as chaves de `seg` e adiciona ou substitui as seguintes.

Finalmente:

```python
    return result
```

Devolve a lista de segmentos agora com oradores identificados.

---

## 4. `pipeline/01_transcribe.py` — O orquestrador (entry point)

Este é o script que o utilizador corre (`python pipeline/01_transcribe.py`).
Ele importa as funções dos outros módulos e coordena o fluxo.

### Imports

```python
import json
import os
import sys

from dotenv import load_dotenv

from audio import extract_audio
from diarize import diarize_audio
from transcribe import transcribe_audio, merge_speaker_labels
```

- `json` — para ler/escrever ficheiros JSON.
- `os` — verificar se ficheiros e pastas existem.
- `sys` — para sair do programa com `sys.exit()` se algo correr mal.
- `load_dotenv` — carrega o ficheiro `.env` que contém as chaves de API
  (neste caso o `HF_TOKEN`).
- As funções que já vimos: `extract_audio`, `diarize_audio`, `transcribe_audio`,
  `merge_speaker_labels`.

### Constantes

```python
INPUT_VIDEO = "input/meeting.mp4"
OUTPUT_FILE = "output/transcript.json"
```

Caminhos de entrada e saída.

### Função `main()`

```python
def main():
```

Toda a lógica está dentro de `main()`, que depois é chamada no fim do ficheiro.

```python
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: {INPUT_VIDEO} not found.")
        sys.exit(1)
```

Se o vídeo não existe, mostra uma mensagem e sai com código de erro 1.

```python
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN is required for diarization. Set it in .env")
        sys.exit(1)
```

Carrega as variáveis de ambiente do ficheiro `.env` e verifica se o
`HF_TOKEN` está definido. Sem ele, a diarização não funciona.

```python
    os.makedirs("output", exist_ok=True)
```

Cria a pasta `output/` se não existir.

```python
    print("→ Extracting audio via ffmpeg...")
    audio_path = extract_audio(INPUT_VIDEO)
    print(f"  Audio saved to {audio_path}")
```

Passo 1: extrair áudio do vídeo.

```python
    print("→ Running WhisperX transcription (large-v2)...")
    segments = transcribe_audio(audio_path)
    print(f"  Transcribed {len(segments)} segments")
```

Passo 2: transcrever o áudio com WhisperX.

```python
    print("→ Running pyannote speaker diarization...")
    diarization = diarize_audio(audio_path, hf_token=hf_token)
    print(f"  Found {len(set(d['speaker'] for d in diarization))} speakers")
```

Passo 3: diarização. A contagem de oradores usa `set()` para obter apenas
os valores únicos (cada orador aparece várias vezes na lista).

```python
    transcript = merge_speaker_labels(segments, diarization)
```

Passo 4: cruzar os segmentos de texto com os oradores.

```python
    with open(OUTPUT_FILE, "w") as f:
        json.dump(transcript, f, indent=2)
```

Escreve o resultado final em `output/transcript.json`, com indentação de 2
espaços para ser legível.

```python
    print(f"✓ Transcript saved to {OUTPUT_FILE}")
```

Confirmação para o utilizador.

### Bloco `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

Este bloco garante que `main()` só é chamada quando o ficheiro é corrido
diretamente (`python pipeline/01_transcribe.py`), e não quando é importado
como módulo (`from pipeline import transcribe`). Isto permite que os testes
importem funções sem executar o pipeline.

---

## Fluxo completo

1. `01_transcribe.py:main()` chama `audio.extract_audio()` → obtém WAV.
2. `main()` chama `transcribe.transcribe_audio()` → obtém segmentos de texto
   com timestamps ao nível da palavra.
3. `main()` chama `diarize.diarize_audio()` → obtém quem falou e quando.
4. `main()` chama `transcribe.merge_speaker_labels()` → combina texto +
   oradores.
5. O resultado é guardado em `output/transcript.json`.

O JSON final parece-se com:
```json
[
  {
    "start": 0.5,
    "end": 3.2,
    "text": "Bom dia a todos",
    "words": [
      {"word": "Bom", "start": 0.5, "end": 0.8},
      {"word": "dia", "start": 0.9, "end": 1.2},
      {"word": "a", "start": 1.3, "end": 1.4},
      {"word": "todos", "start": 1.5, "end": 3.2}
    ],
    "speaker": "SPEAKER_00"
  },
  ...
]
```

Cada entrada é um segmento de fala, com texto, palavras individuais com
timestamps, e o orador que o disse.
