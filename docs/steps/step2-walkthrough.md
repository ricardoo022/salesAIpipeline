# Walkthrough do Step 2 — Extração de Características de Áudio

## O que este step faz

O Step 2 pega o transcript produzido pelo Step 1 (quem disse o quê e quando) e o áudio extraído do vídeo, e calcula métricas de áudio para cada segmento de fala: pitch (tom de voz), energia (volume), taxa de fala (palavras por segundo), pausas e zero crossing rate (uma medida de aspereza/noz do sinal). O resultado é um ficheiro JSON (`output/audio_features.json`) com um objeto por segmento, cada um com os valores numéricos destas características. Estes dados vão alimentar os steps seguintes de emoção (voz e face) e a análise final com a LLM.

## O que cada campo do output significa

Cada segmento no `output/audio_features.json` tem estes campos:

| Campo | O que significa | Por que é importante | Como foi calculado |
|-------|----------------|---------------------|--------------------|
| `speaker` | Quem está a falar (ex: `SPEAKER_00`, `SPEAKER_01`) | Permite agrupar métricas por speaker — essencial para comparar comportamento do vendedor vs cliente | Vem do `transcript.json` (Step 1), que cruza WhisperX com pyannote para atribuir cada segmento a um speaker |
| `start` / `end` | Segundos de início e fim do segmento no vídeo | Sem estas janelas temporais não conseguimos alinhar as métricas com o que foi dito nem com os momentos críticos no relatório final | Vêm diretamente do `transcript.json` |
| `pitch_mean` | Frequência fundamental média da voz no segmento (em Hz) — o "tom" da voz. Vozes grossas têm pitch baixo (~85–180 Hz), vozes finas têm pitch alto (~165–255 Hz) | Tom mais agudo que o normal = excitação, nervosismo ou entusiasmo. Tom anormalmente grave = cansaço ou autoridade. Ajuda a detetar momentos de tensão | `librosa.pyin(segment_audio)` estima o pitch frame a frame; filtramos os frames sem voz (`NaN`); calculamos a média dos restantes |
| `pitch_std` | Desvio padrão do pitch — mede **quanto o tom variou** durante o segmento | Variação alta = pessoa está animada, a fazer perguntas, ou emocional. Variação baixa = tom monocórdico, possivelmente desinteressada ou a ler um guião | `numpy.std()` dos valores de pitch que não eram `NaN` |
| `energy_mean` | Volume médio do segmento (RMS — Root Mean Square), um valor entre 0 (silêncio) e ~1 (máximo) | Energia alta = entusiasmo, ênfase, possivelmente agressividade. Energia baixa = cansaço, desinteresse, tom baixo | `librosa.feature.rms(y=segment_audio)` calcula a energia RMS frame a frame; fazemos a média de todos os frames |
| `speech_rate` | Palavras por segundo — quantas palavras a pessoa disse dividido pela duração do segmento | Falar muito rápido = ansiedade, nervosismo, ou entusiasmo. Falar muito devagar = hesitação, incerteza, ou explicação cautelosa. Ideal em vendas: ritmo moderado com variação | `len(words) / (end - start)` — contamos as palavras no array `words[]` do transcript e dividimos pela duração em segundos |
| `pause_ratio` | Fração do segmento que foi pausa entre palavras (0 = sem pausas, 0.3 = 30% do tempo foi silêncio) | Pausas podem ser estratégicas (dar tempo para pensar) ou sinal de hesitação. Rácio muito alto = nervosismo ou está perdido. Rácio muito baixo = discurso ensaiado ou pressionado | Para cada par de palavras consecutivas, calculamos `(word[i+1].start - word[i].end)`; somamos todos os gaps que são positivos; dividimos pela duração total do segmento |
| `zcr` | Zero Crossing Rate — quantas vezes o sinal de áudio cruzou o zero por unidade de tempo | ZCR alto = som mais "áspero" ou "ruidoso" (fricativas como "s", "f", "ch"). ZCR baixo = som mais suave e periódico (vogais). Pode indicar emoção/agitação na voz vs tom calmo | `librosa.feature.zero_crossing_rate(y=segment_audio)` calcula a taxa frame a frame; fazemos a média |

**Exemplo real de um segmento do output:**
```json
{
  "speaker": "SPEAKER_01",
  "start": 0.251,
  "end": 0.771,
  "pitch_mean": 196.729,
  "pitch_std": 15.1103,
  "energy_mean": 0.0561,
  "speech_rate": 5.7692,
  "pause_ratio": 0.0769,
  "zcr": 0.088
}
```

---

**Data flow do Step 1 (produz os inputs do Step 2):**
```
input/meeting.mp4
  │
  ├──→ audio.py:extract_audio()
  │         ↓
  │    output/audio_temp.wav ─────┐
  │         │                     │
  │         ├──→ transcribe.py   │  (inputs do Step 2)
  │         ├──→ diarize.py      │
  │         └──→ merge_speaker_labels()
  │                   ↓
  │              output/transcript.json ──┘
```

---

## Ficheiro 1: `pipeline/features.py`

Este é o ficheiro que contém a lógica principal — a função que realmente calcula as características de áudio. O `02_audio_features.py` (que veremos a seguir) é apenas o "orquestrador" que chama esta função.

### Imports

```python
import os
import librosa
import numpy as np
```

- **`os`** — biblioteca padrão do Python para interagir com o sistema operativo (ficheiros, diretórios). Aqui é usada para verificar se o ficheiro de áudio existe (`os.path.exists`).
- **`librosa`** — biblioteca especializada em análise de áudio. É o coração deste step. Fornece funções para carregar áudio, estimar pitch, calcular energia, etc.
- **`numpy`** (`np`) — biblioteca para computação numérica (arrays, operações matemáticas). Muitas funções do librosa devolvem arrays numpy, por isso precisamos do numpy para os processar (calcular médias, desvios padrão, filtrar valores).

### Constante

```python
AUDIO_SAMPLE_RATE = 16000
```

Define a taxa de amostragem: **16000 amostras por segundo** (16 kHz). Este é o standard para processamento de áudio para reconhecimento de fala. O áudio original do vídeo pode ter 48 kHz ou outro valor, mas o ffmpeg (no Step 1) converte-o para 16 kHz mono. Esta constante tem de coincidir com o `AUDIO_SAMPLE_RATE` definido em `pipeline/audio.py`.

> **Porquê 16 kHz?** É uma taxa de amostragem suficiente para captar todas as frequências relevantes da voz humana (o espectro da fala está entre ~80 Hz e ~8 kHz). Usar uma taxa mais baixa poupa processamento; usar uma mais alta não traria benefício para análise de voz.

### Função `extract_audio_features(transcript, audio_path)`

```python
def extract_audio_features(transcript: list[dict], audio_path: str) -> list[dict]:
```

Recebe:
- **`transcript`** — uma lista de dicionários, cada um representando um segmento de fala com as chaves `speaker`, `start`, `end`, `words` (entre outras). É o conteúdo do `output/transcript.json` produzido pelo Step 1.
- **`audio_path`** — caminho para o ficheiro WAV (`output/audio_temp.wav`).

Devolve uma nova lista de dicionários — um por segmento — com as métricas calculadas.

#### Validações iniciais

```python
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
if not transcript:
    raise ValueError("Transcript is empty")
```

Se o ficheiro de áudio não existe ou o transcript está vazio, a função levanta um erro — não faz sentido continuar.

#### Carregar o áudio

```python
y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
```

- **`librosa.load()`** carrega o ficheiro WAV para a memória.
- Devolve dois valores:
  - **`y`** — um array numpy com os valores das amostras de áudio (um array 1D de números, cada um representando a amplitude do som num dado instante).
  - **`sr`** — a sample rate real (que será 16000, porque pedimos que carregasse a essa taxa).

Pense em `y` como uma "fotografia" da onda sonora: se o áudio tiver 10 segundos a 16 kHz, `y` terá 160000 amostras (10 × 16000).

#### Loop principal: `for seg in transcript:`

Para cada segmento de fala no transcript:

```python
start_sample = int(seg["start"] * sr)
end_sample = int(seg["end"] * sr)
segment_audio = y[start_sample:end_sample]
```

Converte os tempos de início e fim (em segundos) para índices no array `y`. Por exemplo, se o segmento começa no segundo 2.5 e `sr = 16000`, então `start_sample = 40000`. Depois extrai a parte do array `y` que corresponde a este segmento — `segment_audio` é um "recorte" do áudio completo.

##### Segmento vazio?

```python
if len(segment_audio) == 0:
    pitch_mean = 0.0
    pitch_std = 0.0
    energy_mean = 0.0
    zcr = 0.0
```

Se o segmento é tão curto que não tem amostras (ou o fim é igual ao início), define tudo a zero. É um caso extremo que não deve acontecer com dados normais, mas está aqui para prevenir erros.

##### Extração de pitch (F0) com `librosa.pyin`

```python
f0, _voiced_flag, _voiced_probs = librosa.pyin(
    segment_audio,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
)
voiced_f0 = f0[~np.isnan(f0)]
if len(voiced_f0) > 0:
    pitch_mean = float(np.mean(voiced_f0))
    pitch_std = float(np.std(voiced_f0))
else:
    pitch_mean = 0.0
    pitch_std = 0.0
```

**O que é o pitch?** É a "altura" do som — o que percebemos como tom de voz (grave vs agudo). Uma voz grossa (grave) tem pitch baixo; uma voz fina (aguda) tem pitch alto. Em vendas, o pitch pode indicar entusiasmo ou tensão.

**`librosa.pyin()`** — função que estima a **frequência fundamental (F0)** da voz em cada instante do segmento. É um algoritmo chamado "probabilistic YIN", uma versão melhorada do método YIN clássico para deteção de pitch.

- **`fmin` e `fmax`** — limites de frequência que o algoritmo vai considerar. `librosa.note_to_hz("C2")` converte a nota musical Dó2 (~65 Hz) para hertz; `C7` (~2093 Hz) é o limite superior. Isto cobre praticamente toda a gama de vozes humanas (homens: ~85–180 Hz, mulheres: ~165–255 Hz).
- Devolve **três arrays**:
  - `f0` — estimativas de pitch para cada frame (ou `NaN` se não foi possível estimar — silêncio, consoantes, ruído).
  - `_voiced_flag` — booleano indicando se o frame tem voz ou não.
  - `_voiced_probs` — probabilidade de ser voz.
- Ignoramos os dois últimos (daí o `_`), só nos interessa `f0`.

**Tratamento dos `NaN`:** `f0` contém `NaN` nos frames onde não há voz (pausas, sons não-vozeados como "sss" ou "shh"). Criamos `voiced_f0` que só contém os valores que **não** são `NaN` (`~np.isnan(f0)` significa "onde não é NaN"). Depois calculamos a média (`np.mean`) — pitch médio do falante neste segmento — e o desvio padrão (`np.std`) — variação do tom. Um desvio padrão alto significa que a pessoa variou muito o tom (talvez animada ou a fazer perguntas); baixo significa tom monocórdico.

Se não houver nenhum frame com voz (segmento de silêncio total), pitch fica a zero.

##### Extração de energia com `librosa.feature.rms`

```python
rms = librosa.feature.rms(y=segment_audio)
energy_mean = float(np.mean(rms))
```

**O que é a energia?** É essencialmente o **volume** ou **intensidade** do som — a amplitude da onda sonora. `RMS` (Root Mean Square) é uma forma de calcular a energia média: eleva os valores ao quadrado, calcula a média, tira a raiz quadrada. O resultado é um número que reflete o volume da fala.

Em contexto de vendas: energia alta pode significar entusiasmo ou agressividade; energia baixa pode significar cansaço ou desinteresse.

`librosa.feature.rms()` devolve um array 1D com valores RMS para cada frame. Tiramos a média para obter um valor único para todo o segmento.

##### Zero Crossing Rate (ZCR)

```python
zcr_array = librosa.feature.zero_crossing_rate(y=segment_audio)
zcr = float(np.mean(zcr_array))
```

**O que é o ZCR?** É a **taxa de passagens por zero** — quantas vezes o sinal de áudio cruza o valor zero (muda de positivo para negativo ou vice-versa) por unidade de tempo.

**Para que serve?** Sons mais "ruidosos" ou "ásperos" (como consoantes fricativas: "f", "s", "ch") cruzam o zero muitas vezes; sons mais suaves e periódicos (vogais) cruzam poucas vezes. Em vendas, ZCR pode ajudar a detetar:
- Emoção/agitação na voz (ZCR mais alto)
- Tom calmo e controlado (ZCR mais baixo)
- Momentos de ênfase ou stress

##### Taxa de fala (speech rate)

```python
words = seg.get("words", [])
duration = seg["end"] - seg["start"]
if duration > 0 and words:
    speech_rate = len(words) / duration
else:
    speech_rate = 0.0
```

**`seg.get("words", [])`** — obtém a lista de palavras do segmento (com timestamps). O `get` com valor padrão `[]` evita erro se a chave não existir.

**`duration = seg["end"] - seg["start"]`** — duração do segmento em segundos.

**`speech_rate = len(words) / duration`** — número de palavras dividido pela duração = **palavras por segundo**. Uma taxa de fala alta pode indicar ansiedade ou entusiasmo; baixa pode indicar hesitação ou explicação cautelosa. Em vendas, a taxa ideal depende do contexto — explicar um produto devagar transmite confiança; falar rápido pode soar nervoso.

Se a duração for zero (`duration > 0`) ou não houver palavras, a taxa é zero.

##### Rácio de pausas (pause ratio)

```python
if duration > 0 and len(words) > 1:
    gaps = 0.0
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap > 0:
            gaps += gap
    pause_ratio = gaps / duration
else:
    pause_ratio = 0.0
```

**Lógica:** percorre as palavras aos pares (palavra i e palavra i+1), calcula o tempo entre o fim de uma e o início da seguinte — isso é o "gap" ou pausa. Soma todos os gaps. Depois divide pela duração total do segmento para obter a **fração do tempo que foi pausa**.

Exemplo: se o segmento tem 10 segundos e os gaps entre palavras somam 2 segundos, `pause_ratio = 0.2` (20% do tempo foi silêncio entre palavras).

**Porquê `len(words) > 1`?** Se só há uma palavra, não há gaps para calcular. Se `duration > 0` é falso (segmento de duração zero) ou não há palavras suficientes, o rácio é zero.

Em vendas: pausas podem ser estratégicas (dar tempo para o cliente processar) ou sinal de hesitação (não saber o que dizer). Um rácio de pausas muito alto pode indicar nervosismo; muito baixo pode soar a discurso ensaiado.

##### Construir o dicionário de resultado

```python
result.append({
    "speaker": seg["speaker"],
    "start": seg["start"],
    "end": seg["end"],
    "pitch_mean": round(pitch_mean, 4),
    "pitch_std": round(pitch_std, 4),
    "energy_mean": round(energy_mean, 4),
    "speech_rate": round(speech_rate, 4),
    "pause_ratio": round(pause_ratio, 4),
    "zcr": round(zcr, 4),
})
```

Mantém os campos originais (`speaker`, `start`, `end`) e adiciona os cinco valores calculados, todos arredondados a 4 casas decimais.

No final, a função devolve a lista `result`.

---

## Ficheiro 2: `pipeline/02_audio_features.py`

Este é o script que **orquestra** a extração — a "entrada" da pipeline para o Step 2. É o que é executado quando corres `python pipeline/02_audio_features.py` (ou é chamado pelo `run.py`).

### Shebang e docstring

```python
#!/usr/bin/env python3
"""Step 2: Extract audio features from transcript segments.

...
"""
```

- **`#!/usr/bin/env python3`** — shebang que permite executar o ficheiro diretamente como script (com `./02_audio_features.py`), embora na prática seja sempre invocado com `python pipeline/02_audio_features.py`.
- **Docstring** — explica o propósito, o que consome e o que produz.

### Imports

```python
import json
import os
import sys

from features import extract_audio_features
```

- **`json`** — para ler o transcript (JSON) e escrever o resultado (JSON).
- **`os`** — para verificar se ficheiros existem (`os.path.exists`) e criar diretórios (`os.makedirs`).
- **`sys`** — para sair do programa com erro (`sys.exit(1)`).
- **`from features import extract_audio_features`** — importa a função que acabámos de explicar, definida em `pipeline/features.py`. Nota: o import é relativo porque o Python adiciona a diretoria do script ao `sys.path` quando o executamos diretamente.

### Constantes

```python
TRANSCRIPT_FILE = "output/transcript.json"
AUDIO_FILE = "output/audio_temp.wav"
OUTPUT_FILE = "output/audio_features.json"
```

Caminhos dos ficheiros de input e output, relativos à raiz do projeto.

### Função `main()`

```python
def main():
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"ERROR: {TRANSCRIPT_FILE} not found. Run step 1 first.")
        sys.exit(1)
```

**Verificação 1:** se o ficheiro do transcript não existe, imprime uma mensagem de erro clara e sai com código 1 (indicando erro). Isto é importante porque o Step 2 depende do output do Step 1.

```python
    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Run step 1 first.")
        sys.exit(1)
```

**Verificação 2:** o ficheiro de áudio também tem de existir. O `run.py` ou o Step 1 garante que este ficheiro existe antes de avançar.

```python
    with open(TRANSCRIPT_FILE) as f:
        transcript = json.load(f)
```

Abre o ficheiro JSON do transcript e carrega-o para a variável `transcript` (uma lista de dicionários). O `with` garante que o ficheiro é fechado automaticamente, mesmo que ocorra um erro.

```python
    print("→ Extracting audio features (pitch, energy, speech rate, pauses, ZCR)...")
    features = extract_audio_features(transcript, AUDIO_FILE)
```

Chama a função que vimos em `features.py`, passando o transcript e o caminho do áudio. O resultado fica em `features` — uma lista de dicionários com as métricas calculadas.

```python
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(features, f, indent=2)
```

Cria a diretoria `output/` se não existir (`exist_ok=True` significa "não dá erro se já existir"). Depois abre o ficheiro de output para escrita (`"w"`) e escreve a lista como JSON formatado com indentação de 2 espaços — fica legível para um humano abrir e inspecionar.

```python
    print(f"✓ Audio features saved to {OUTPUT_FILE} ({len(features)} segments)")
```

Mensagem de sucesso, indicando quantos segmentos foram processados.

### Guarda de execução

```python
if __name__ == "__main__":
    main()
```

**`if __name__ == "__main__"`** — esta condição é `True` apenas quando o ficheiro é executado diretamente (como `python pipeline/02_audio_features.py`). Se for importado por outro módulo (como nos testes), `__name__` tem outro valor e o `main()` não é chamado. Isto permite que as funções sejam importadas e testadas sem executar o script.

---

## Ficheiro 3: `pipeline/audio.py` (apenas a constante)

```python
AUDIO_SAMPLE_RATE = 16000
```

Este ficheiro contém a função `extract_audio()` que extrai o áudio do vídeo com ffmpeg. Para o Step 2, a única parte relevante é a constante **`AUDIO_SAMPLE_RATE = 16000`**, que é o valor usado no Step 1 para converter o áudio do vídeo para 16 kHz.

Quando o ficheiro `output/audio_temp.wav` está a ser gerado, o ffmpeg usa este valor para definir a taxa de amostragem. O Step 2, por sua vez, carrega o áudio com `librosa.load(audio_path, sr=16000)` — e é fundamental que ambas as partes usem o mesmo valor, senão os timestamps do transcript (em segundos) não corresponderiam corretamente ao áudio.

> **Resumo:** o `AUDIO_SAMPLE_RATE` nos dois ficheiros tem de ser o mesmo. Se um dia mudar para 22050 Hz (outro valor comum), tem de ser alterado em ambos os sítios.

---

## Visão Geral do Fluxo

```
output/transcript.json ──┐
                         ├──→ pipeline/features.py:extract_audio_features()
output/audio_temp.wav  ──┘
                                 ↓
                         output/audio_features.json
```

1. O `02_audio_features.py` lê o transcript e verifica se o áudio existe.
2. Chama `extract_audio_features(transcript, audio_path)`.
3. A função carrega o áudio com `librosa.load`, itera por cada segmento do transcript, extrai o áudio desse segmento e calcula:
   - **Pitch** (tom de voz) — com `librosa.pyin()`
   - **Energia** (volume) — com `librosa.feature.rms()`
   - **Zero Crossing Rate** (aspereza) — com `librosa.feature.zero_crossing_rate()`
   - **Speech rate** (palavras por segundo) — a partir das palavras com timestamps
   - **Pause ratio** (fração de pausas) — a partir dos gaps entre palavras
4. Grava o resultado como `output/audio_features.json`.

Este JSON será consumido pelo Step 3 (emotion_voice) que vai associar emoções a cada segmento com base nas mesmas janelas temporais.
