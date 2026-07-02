# Walkthrough do Step 3 — Emoção na Voz

## O que este step faz

O Step 3 pega nos segmentos de áudio produzidos pelo Step 2 (quem disse o quê, quando, e com que características acústicas) e classifica a **emoção na voz** de cada segmento usando um modelo de deep learning pré-treinado. O modelo `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` analisa o sinal de áudio bruto e produz três valores contínuos entre 0 e 1: **valence** (quão positivo/negativo), **arousal** (quão calmo/excitado) e **dominance** (quão submisso/dominante). Segmentos mais longos que 15 segundos são divididos em chunks, processados individualmente, e os resultados são averaged de volta ao nível do segmento. O output é `output/voice_emotion.json` — um array com um objeto por segmento, cada um com `{speaker, start, end, valence, arousal, dominance}`.

## O que cada campo do output significa

Cada segmento no `output/voice_emotion.json` tem estes campos:

| Campo | O que significa | Por que é importante em vendas | Como foi calculado |
|-------|----------------|-------------------------------|--------------------|
| `speaker` | Quem está a falar (ex: `SPEAKER_00`, `SPEAKER_01`) | Permite separar as emoções do vendedor das do cliente — essencial para o relatório comparativo | Vem do `audio_features.json` (Step 2), que por sua vez vem do `transcript.json` (Step 1) |
| `start` / `end` | Segundos de início e fim do segmento no vídeo | Permite alinhar cada classificação emocional com o momento exato do vídeo e com o que foi dito | Vêm diretamente do `audio_features.json` |
| `valence` | Quão positiva ou negativa a voz soou (0 = muito negativa, 1 = muito positiva) | Valence baixo = cliente desconfortável, céptico ou insatisfeito. Valence alto = entusiasmo, concordância, interesse. Um cliente que diz "sim" com valence baixo não está realmente convencido | O modelo wav2vec2 produz logits; aplicamos `torch.sigmoid()` para obter valores entre 0 e 1 |
| `arousal` | Quão calmo ou excitado o falante soou (0 = calmo/passivo, 1 = muito excitado) | Arousal baixo + valence baixo = zona de perigo (cliente desligado, desinteressado). Arousalalto + valence alto = engajamento e entusiasmo. Um aumento súbito de arousal pode indicar objeção forte | Mesma pipeline: sigmoid nos logits do modelo |
| `dominance` | Quão dominante ou submisso o falante soou (0 = submisso/passivo, 1 = dominante/assertivo) | Um shift de dominance no cliente = pushback ou objeção a formar-se. Dominance alto no vendedor = a controlar a conversa (bom ou mau, depende do contexto). Em vendas consultivas, queremos dominance equilibrado | Mesma pipeline: sigmoid nos logits do modelo |

> **Porquê estes três e não "feliz/triste/zangado"?** O modelo VAD (valence-arousal-dominance) é uma representação dimensional da emoção usada em psicologia. Em vez de categorias discretas, captura o espaço emocional contínuo — muito mais útil para detetar variações subtis que umaSell consultiva precisa. Uma frase pode não ser claramente "triste" nem "zangada", mas ter valence baixo e arousal médio — sinal de cepticismo.

**Exemplo de um segmento do output:**
```json
{
  "speaker": "SPEAKER_01",
  "start": 12.4,
  "end": 18.1,
  "valence": 0.3127,
  "arousal": 0.2241,
  "dominance": 0.4103
}
```

---

## O modelo: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`

Antes de mergulhar no código, vale a pena perceber o que é este modelo.

- **`wav2vec2`** — arquitetura de deep learning da Meta/Facebook, pré-treinada em grandes quantidades de áudio não rotulado para aprender representações de fala. Funciona diretamente sobre o sinal de áudio bruto (waveform), sem precisar de features manuais como MFCCs.
- **`large-robust-12-ft-emotion-msp-dim`** — a variante específica:
  - **large** — versão maior (mais parâmetros, mais precisa, mais lenta)
  - **robust** — treinada com dados ruidosos para ser resistente a condições reais
  - **12-ft** — "12-layer fine-tuned" — o modelo base foi fine-tuned para a tarefa de emoção
  - **emotion-msp-dim** — fine-tuned no dataset MSP-Dim (Munich Dimensional Speech Emotion), que tem anotações VAD contínuas
- **Input:** um array numpy com o sinal de áudio a 16 kHz
- **Output:** 3 logits (valence, arousal, dominance) — aplicamos sigmoid para obter probabilidades 0–1

O modelo é **downloaded uma vez** e **guardado em cache** na diretoria `models/` do projeto. Nas execuções seguintes, o `transformers` usa a cache local em vez de fazer download novamente (~1.3 GB).

---

## Ficheiro 1: `pipeline/emotion_voice.py`

Este é o ficheiro com a lógica principal. Contém três funções: uma para carregar o modelo, uma para fazer uma previsão num chunk de áudio, e a função principal que processa todos os segmentos.

### Imports

```python
import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
```

- **`os`** — para verificar se o ficheiro de áudio existe e criar a diretoria de cache do modelo.
- **`librosa`** — para carregar o ficheiro WAV (igual ao Step 2). Carrega o áudio como um array numpy à taxa de 16 kHz.
- **`numpy`** (`np`) — para operações com arrays (média dos chunks, conversão de tipos).
- **`torch`** — PyTorch, necessário para desativar gradients (`torch.no_grad()`) e aplicar a sigmoid (`torch.sigmoid()`).
- **`transformers`** — biblioteca da HuggingFace que fornece o modelo e o feature extractor. O `Wav2Vec2FeatureExtractor` prepara o áudio bruto num formato que o modelo entende; o `Wav2Vec2ForSequenceClassification` é o próprio modelo que produz os logits de emoção.

### Import path duality

```python
try:
    from pipeline.audio import AUDIO_SAMPLE_RATE
except ImportError:
    from audio import AUDIO_SAMPLE_RATE
```

Este padrão já foi visto nos steps anteriores. O `try` cobre o caso em que o módulo é importado como package (nos testes: `from pipeline.emotion_voice import ...`); o `except` cobre o caso em que é importado diretamente (na CLI: `from emotion_voice import ...`, porque o Python adiciona a diretoria do script ao `sys.path`). Em ambos os casos, obtemos `AUDIO_SAMPLE_RATE = 16000` definido em `pipeline/audio.py`.

### Constantes

```python
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_CACHE_DIR = "models"
MAX_CHUNK_DURATION = 15
```

- **`MODEL_NAME`** — identificador do modelo no HuggingFace Hub. A HuggingFace usa este string para fazer download e guardar em cache.
- **`MODEL_CACHE_DIR = "models"`** — diretoria onde os pesos do modelo são guardados. Isto garante que o modelo (~1.3 GB) é downloaded apenas uma vez e reutilizado em execuções futuras.
- **`MAX_CHUNK_DURATION = 15`** — duração máxima em segundos de cada chunk enviado ao modelo. O modelo wav2vec2 processa áudio de comprimento variável, mas segmentos muito longos (minutos) consomem muita memória e perdem precisão. 15 segundos é um compromise entre granularidade e estabilidade.

### Função `_load_model()`

```python
def _load_model(cache_dir: str = MODEL_CACHE_DIR):
    """Load the emotion model and feature extractor, cached in models/."""
    os.makedirs(cache_dir, exist_ok=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir
    )
    model.eval()
    return feature_extractor, model
```

Esta é uma função privada (o `_` no início é uma convenção Python para "uso interno do módulo"). Faz três coisas:

1. **`os.makedirs(cache_dir, exist_ok=True)`** — garante que a diretoria `models/` existe (cria-a se necessário, não dá erro se já existir).

2. **Carrega o feature extractor** — o `Wav2Vec2FeatureExtractor` é o componente que prepara o áudio bruto (array numpy de amostras) no formato que o modelo espera (normalização, padding, conversão para tensor). O `from_pretrained` descarrega-o do HuggingFace na primeira execução e usa a cache local nas seguintes.

3. **Carrega o modelo** — o `Wav2Vec2ForSequenceClassification` é a rede neural wav2vec2-large com uma cabeça de classificação de 3 outputs (valence, arousal, dominance). O `from_pretrained` descarrega os pesos (~1.3 GB) na primeira execução.

4. **`model.eval()`** — coloca o modelo em "modo de avaliação" (inference). Isto desativa dropout e other comportamentos de treino que não fazem sentido para previsão.

Devolve o par `(feature_extractor, model)` para ser usado pela função de previsão.

### Função `_predict_chunk()`

```python
def _predict_chunk(audio_chunk: np.ndarray, feature_extractor, model) -> np.ndarray:
    """Run emotion model on a single audio chunk, returning [valence, arousal, dominance]."""
    inputs = feature_extractor(
        audio_chunk, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).numpy()[0]
    return probs
```

Esta função recebe um chunk de áudio (array numpy) e devolve um array com 3 valores: `[valence, arousal, dominance]`.

1. **`feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt")`** — transforma o array numpy num dicionário de tensors PyTorch (`"pt"` = PyTorch). O feature extractor normaliza o áudio e prepara-o no formato exato que o modelo foi treinado para receber.

2. **`with torch.no_grad():`** — desliga o cálculo de gradients. Como estamos a fazer previsão (não treino), não precisamos de gradientes e poupa muita memória e tempo.

3. **`outputs = model(**inputs)`** — corre o modelo. O `**inputs` espalha o dicionário em argumentos nomeados (ex: `model(input_values=..., attention_mask=...)`). O output é um objeto com o atributo `.logits`.

4. **`torch.sigmoid(outputs.logits)`** — o modelo devolve **logits** (valores reais sem restrição). A `sigmoid` converte-os para o intervalo [0, 1]:
   - sigmoid(x) = 1 / (1 + e^(-x))
   - Logit 0 → sigmoid 0.5 (neutro)
   - Logit positivo → sigmoid > 0.5 (acima da média)
   - Logit negativo → sigmoid < 0.5 (abaixo da média)

5. **`.numpy()[0]`** — converte o tensor PyTorch para array numpy e extrai a primeira (e única) linha (index 0), porque o modelo processa um chunk de cada vez e devolve um array 2D com shape `(1, 3)`.

O resultado é algo como `array([0.3127, 0.2241, 0.4103])` — valence, arousal, dominance.

### Função `extract_voice_emotion()`

```python
def extract_voice_emotion(
    segments: list[dict],
    audio_path: str,
    model_path: str = None,
) -> list[dict]:
```

Esta é a função principal, análoga à `extract_audio_features()` do Step 2. Recebe:
- **`segments`** — lista de segmentos (o conteúdo do `audio_features.json`), cada um com `speaker`, `start`, `end`.
- **`audio_path`** — caminho para o ficheiro WAV (`output/audio_temp.wav`).
- **`model_path`** — caminho opcional para a cache do modelo (default: `"models"`). Útil nos testes para usar uma cache temporária.

Devolve uma nova lista com `speaker`, `start`, `end` + `valence`, `arousal`, `dominance`.

#### Validações iniciais

```python
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
if not segments:
    raise ValueError("Segments list is empty")
```

Mesma estrutura de validação do Step 2: se o áudio não existe ou não há segmentos, para já com erro.

#### Carregar áudio e modelo

```python
y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
cache_dir = model_path or MODEL_CACHE_DIR
feature_extractor, model = _load_model(cache_dir)
max_chunk_samples = MAX_CHUNK_DURATION * AUDIO_SAMPLE_RATE
```

- **`y, sr = librosa.load(...)`** — carrega o áudio completo como array numpy `y` à taxa `sr` (16000). Igual ao Step 2.
- **`cache_dir = model_path or MODEL_CACHE_DIR`** — se foi passado um `model_path` (nos testes), usa-o; senão usa `"models"`.
- **`_load_model(cache_dir)`** — carrega o feature extractor e o modelo (com cache).
- **`max_chunk_samples = 15 * 16000 = 240000`** — número máximo de amostras por chunk. 15 segundos × 16000 amostras/segundo.

#### Loop principal: `for seg in segments:`

Para cada segmento:

```python
start_sample = int(seg["start"] * sr)
end_sample = int(seg["end"] * sr)
segment_audio = y[start_sample:end_sample]
```

Igual ao Step 2: converte os tempos (segundos) em índices do array e extrai o recorte de áudio correspondente a este segmento.

##### Segmento vazio?

```python
if len(segment_audio) == 0:
    result.append({
        "speaker": seg["speaker"],
        "start": seg["start"],
        "end": seg["end"],
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
    })
    continue
```

Se o segmento não tem amostras (caso extremo: `start == end`), regista o segmento com valores 0.0 e salta para o próximo. Não faria sentido enviar um array vazio ao modelo.

##### Segmento longo: dividir em chunks

```python
if len(segment_audio) > max_chunk_samples:
    chunk_probs = []
    for chunk_start in range(0, len(segment_audio), max_chunk_samples):
        chunk_end = min(chunk_start + max_chunk_samples, len(segment_audio))
        chunk = segment_audio[chunk_start:chunk_end]
        chunk_probs.append(_predict_chunk(chunk, feature_extractor, model))
    probs = np.mean(chunk_probs, axis=0)
```

- **`range(0, len(segment_audio), max_chunk_samples)`** — gera os índices de início de cada chunk (0, 240000, 480000, ...). Cada iteração avança `max_chunk_samples` amostras.
- **`chunk_end = min(chunk_start + max_chunk_samples, len(segment_audio))`** — o fim do chunk é o início + 15 segundos, **ou** o fim do segmento (o último chunk pode ser mais curto). O `min` garante que não ultrapassamos o segmento.
- **`chunk_probs.append(_predict_chunk(...))`** — corre o modelo em cada chunk e guarda os 3 valores.
- **`np.mean(chunk_probs, axis=0)`** — calcula a média de todos os chunks, eixo 0 (média por coluna: média de todos os valences, média de todos os arousals, média de todos os dominances). O resultado é um único array `[valence, arousal, dominance]` que representa o segmento inteiro.

> **Por que é que averaging funciona?** Se um segmento de 30 segundos tem valence 0.2 nos primeiros 15 segundos e 0.8 nos últimos 15, a média é 0.5. Isto perde a granularidade temporal intra-segmento, mas o Step 5 (LLM analysis) não precisa dessa granularidade — quer uma leitura emocional por segmento (turno de fala). Para análise temporal mais fina, o gráfico de engagement no relatório usa múltiplos segmentos curtos que já vêm da diarização.

##### Segmento normal: previsão direta

```python
else:
    probs = _predict_chunk(segment_audio, feature_extractor, model)
```

Se o segmento tem 15 segundos ou menos, corre o modelo uma vez sobre o segmento inteiro. Sem chunks, sem averaging.

##### Construir o dicionário de resultado

```python
result.append({
    "speaker": seg["speaker"],
    "start": seg["start"],
    "end": seg["end"],
    "valence": round(float(probs[0]), 4),
    "arousal": round(float(probs[1]), 4),
    "dominance": round(float(probs[2]), 4),
})
```

Mantém os campos originais (`speaker`, `start`, `end`) e adiciona os três valores emocionais, arredondados a 4 casas decimais. O índice 0 é valence, 1 é arousal, 2 é dominance — esta ordem é definida pelo modelo e está documentada na HuggingFace model card.

No final, a função devolve a lista `result`.

---

## Ficheiro 2: `pipeline/03_emotion_voice.py`

Este é o orquestrador — o script que o utilizador corre diretamente ou que é chamado pelo `run.py`.

### Imports

```python
import json
import os
import sys
```

- **`json`** — para ler `audio_features.json` e escrever `voice_emotion.json`.
- **`os`** — para verificar se os ficheiros de input existem e criar a diretoria de output.
- **`sys`** — para sair com código de erro (`sys.exit(1)`).

> **Nota:** O import `from emotion_voice import extract_voice_emotion` **não está no topo do ficheiro** — está dentro da função `main()`, após as validações. Isto é propositado: o `emotion_voice.py` importa `torch` e `transformers` ao ser carregado, e esses imports são pesados (~20 segundos). Ao adiar o import para dentro da função, as validações de input (`os.path.exists`) executam instantaneamente — se um ficheiro falta, o utilizador recebe o erro em milissegundos em vez de esperar 20 segundos pelo carregamento do PyTorch. Este padrão é igual ao que o `01_transcribe.py` faz com o pyannote.

### Constantes

```python
SEGMENTS_FILE = "output/audio_features.json"
AUDIO_FILE = "output/audio_temp.wav"
OUTPUT_FILE = "output/voice_emotion.json"
```

Caminhos de input e output:
- **`SEGMENTS_FILE`** — os segmentos do Step 2 (com speaker, start, end, e features acústicas). O Step 3 só precisa de `speaker`, `start`, `end` — as features acústicas são ignoradas, mas o ficheiro é consumido tal-and-tal porque já tem a estrutura certa.
- **`AUDIO_FILE`** — o ficheiro WAV extraído no Step 1.
- **`OUTPUT_FILE`** — onde guardar o resultado.

### Função `main()`

```python
def main():
    if not os.path.exists(SEGMENTS_FILE):
        print(f"ERROR: {SEGMENTS_FILE} not found. Run step 2 first.")
        sys.exit(1)

    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Run step 1 first.")
        sys.exit(1)
```

**Validação de inputs.** Se os ficheiros de entrada não existem, mostra uma mensagem de erro clara e sai com código 1. Estes checks acontecem antes de importar o `emotion_voice` (e, portanto, antes de carregar o PyTorch) — a validação é instantânea.

```python
    from emotion_voice import extract_voice_emotion
```

**Import deferido.** Só depois das validações é que importamos a função — e, consequentemente, carregamos o PyTorch e o transformers.

```python
    with open(SEGMENTS_FILE) as f:
        segments = json.load(f)
```

Carrega o `audio_features.json` para a variável `segments` (uma lista de dicionários).

```python
    print("→ Extracting voice emotion (valence, arousal, dominance) from audio segments...")
    emotions = extract_voice_emotion(segments, AUDIO_FILE)
```

Chama a função principal do módulo. Isto é onde o trabalho pesado acontece: carrega o áudio, carrega o modelo wav2vec2 (ou usa a cache em `models/`), e processa cada segmento.

```python
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)
```

Garante que a diretoria `output/` existe e escreve o resultado em `voice_emotion.json` com indentação de 2 espaços.

```python
    print(f"✓ Voice emotion saved to {OUTPUT_FILE} ({len(emotions)} segments)")
```

Confirmação com o número de segmentos processados.

### Bloco `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

Garante que `main()` só é chamada quando o ficheiro é executado diretamente, não quando é importado (como nos testes).

---

## Visão Geral do Fluxo

```
output/audio_features.json ──┐
                             ├──→ pipeline/emotion_voice.py:extract_voice_emotion()
output/audio_temp.wav  ──────┤            │
                             │            ├── _load_model() → wav2vec2 (cached in models/)
                             │            ├── for each segment:
                             │            │     ├── extract audio slice
                             │            │     ├── if > 15s: split into chunks, predict each, average
                             │            │     └── else: predict directly
                             │            └── {speaker, start, end, valence, arousal, dominance}
                             ↓
                     output/voice_emotion.json
```

1. O `03_emotion_voice.py` verifica se `audio_features.json` e `audio_temp.wav` existem.
2. Importa `extract_voice_emotion` (que carrega PyTorch + transformers + o modelo audeering).
3. Chama `extract_voice_emotion(segments, audio_path)`.
4. A função:
   - Carrega o áudio completo com `librosa.load`.
   - Carrega o modelo wav2vec2 com cache em `models/`.
   - Para cada segmento:
     - Extrai o recorte de áudio correspondente.
     - Se for maior que 15 segundos, divide em chunks de 15s, prevê cada um, e faz a média.
     - Caso contrário, prevê diretamente.
     - Aplica `sigmoid` aos logits para obter valence, arousal, dominance em [0, 1].
   - Devolve a lista de segmentos com as três emoções.
5. O resultado é gravado em `output/voice_emotion.json`.

O JSON final tem este aspeto:
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 12.4,
    "end": 18.1,
    "valence": 0.3127,
    "arousal": 0.2241,
    "dominance": 0.4103
  },
  {
    "speaker": "SPEAKER_01",
    "start": 18.3,
    "end": 25.7,
    "valence": 0.5821,
    "arousal": 0.7134,
    "dominance": 0.6298
  },
  ...
]
```

---

## Interpretação em contexto de vendas

O relatório final (Step 6) usa estes dados para:

1. **Engagement timeline** — um gráfico Chart.js com três linhas:
   - Prospect Valence (azul) — quão positivo o cliente está ao longo do tempo
   - Prospect Arousal (laranja) — quão energizado/excitado o cliente está
   - Rep Arousal (cinza, "Rep Energy" no UI) — energia do vendedor

2. **Deteção de momentos críticos** — o gráfico tem marcadores verticais em momentos críticos identificados pela LLM (Step 5). Cada marcador corresponde a um timestamp no vídeo.

3. **Análise comparativa multimodal** — o Step 5 combina o voice emotion do Step 3 com o facial emotion do Step 4 e com as features acústicas do Step 2 para produzir insights que a análise transcript-only não consegue. Por exemplo:
   - Cliente diz "parece interessante" (texto positivo) com valence baixo e arousal baixo (voz) e expressão facial neutra → a LLM multimodal deteta que o cliente não está convencido, enquanto a transcript-only regista "interessante" como positivo.
   - Cliente diz "não sei se faz sentido" com dominance alto e arousal alto → objeção forte a formar-se, não apenas dúvida.

**Padrões a observar:**
- **Arousal baixo + valence baixo** = plano, desengajado — zona de perigo (o cliente está a desligar-se)
- **Arousal alto + valence alto** = excitado, engajado — momento positivo
- **Shift de dominance no cliente** = pushback ou objeção forte a formar-se
- **Valence a descer ao longo da reunião** = o cliente está a perder interesse progressivamente

---

## Bugs encontrados e corrigidos

A primeira versão do Step 3 **passava em todos os testes mas produzia output inútil** — uma linha plana de emoção que falhava o critério de sucesso da spec ("the engagement timeline shows a meaningful signal, not a flat line"). Os testes unitários estavam corretos para a API que tínhamos definido, mas a API estava errada: os testes validavam o comportamento do mock, não o comportamento do modelo real. Os bugs só foram descobertos quando validámos o output real com estatística. Esta secção documenta os dois bugs, como foram encontrados, como foram corrigidos, e por que isso é crítico para o projeto.

### Como os bugs foram descobertos

Depois de os 10 testes passarem e o `output/voice_emotion.json` ser gerado (197 segmentos), corremos uma **rúbrica de validação estatística** sobre o output real — não sobre mocks, sobre os valores que o relatório final iria mostrar. Quatro checks:

| Check | O que "bom" parece | O que obtivemos | Veredicto |
|---|---|---|---|
| **Discriminação** — o modelo distingue segmentos? | VAD espalhado por [0,1], std ≥ ~0.1 | std ≈ 0.003, range ≈ 0.02; 100% dos valores em 0.45–0.55 | ❌ Linha plana |
| **Correlação acústica** — arousal deve跟踪 energy/speech_rate | r > ~0.2 | r ≈ 0 (−0.04 a −0.07) | ❌ Sem sinal |
| **Sensibilidade ao input** — áudio loud vs quiet → arousal diferente | outputs claramente diferentes | loud ≈ quiet (idêntico a 4 casas decimais) | ❌ Não responde |
| **Separação por speaker** — vendedor e cliente têm perfis diferentes | Perfis distintos por speaker | Todos os speakers em 0.50 ± 0.003 | ❌ Sem separação |

Qualquer um destes a falhar seria suficiente para rejeitar o output. Os quatro falharam — a linha plana era o exato failure mode que a spec alertou.

A lição: **testes unitários com mocks validam a plumbing, não o modelo.** Os testes confirmavam que a função chamava o modelo, lia `.logits`, aplicava sigmoid, e construía o dict — tudo correto para a API que tínhamos definido. Mas a API estava errada: o modelo real não tem `.logits`, não produz logits de classificação, e sigmoid é a transformação errada. Só uma validação estatística sobre o output real consegue apanhar isto.

### Bug 1: Classe de modelo errada → cabeça random-initialized

**O problema:** O modelo `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` foi carregado com `Wav2Vec2ForSequenceClassification` — a classe stock do `transformers` para classificação de áudio. Mas este modelo **não** é um classificador stock. O `config.json` do modelo diz explicitamente:

```json
"architectures": ["Wav2Vec2ForSpeechClassification"],
"problem_type": "regression"
```

A classe `Wav2Vec2ForSpeechClassification` era parte do `transformers` em versões antigas, mas **foi removida** em versões recentes (a versão instalada é 4.57.6). A `Wav2Vec2ForSequenceClassification` espera uma cabeça com nomes `classifier.weight` e `classifier.bias` (uma camada linear única). O audeering model tem uma cabeça com nomes diferentes — `classifier.dense.weight` + `classifier.out_proj.weight` ( MLP de duas camadas com mean-pooling). O resultado: o `from_pretrained` não encontra os pesos da cabeça, **silenciosamente inicializa-os aleatoriamente**, e imprime um warning que é fácil ignorar:

```
Some weights of Wav2Vec2ForSequenceClassification were not initialized from the
model checkpoint... ['classifier.bias', 'classifier.weight', 'projector.bias',
'projector.weight']
You should probably TRAIN this model on a down-stream task...
```

Com pesos aleatórios, o modelo produz logits ≈ 0 → `sigmoid(0) = 0.5` → todos os valores em 0.5, sem variação. Cada segmento "recebia" a mesma classificação emocional, independentemente do áudio.

**Como foi corrigido:** Reconstruímos a classe `Wav2Vec2ForSpeechClassification` dentro de `pipeline/emotion_voice.py`, com a arquitetura exata que os pesos esperam:

```python
class Wav2Vec2ClassificationHead(nn.Module):
    """Cabeça guardada no checkpoint como classifier.dense + classifier.out_proj."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, x):
        x = self.dropout(x); x = self.dense(x); x = torch.tanh(x)
        x = self.dropout(x); x = self.out_proj(x); return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Reconstrução da cabeça audeering: mean-pool hidden states -> MLP -> regression."""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)
        self.init_weights()
    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=None)
        hidden_states = outputs[0]
        pooled = hidden_states.mean(dim=1)  # pooling_mode: "mean" (do config)
        return self.classifier(pooled)      # regression, NÃO classificação
```

Ao usar nomes de camadas que correspondem aos pesos saved (`classifier.dense`, `classifier.out_proj`), o `from_pretrained` carrega agora os pesos treinados em vez de os inicializar aleatoriamente. Os warnings de random-init desapareceram; as normas dos pesos são reais (21.1 e 1.05), não random.

### Bug 2: sigmoid aplicada a um modelo de regressão

**O problema:** O `problem_type` do `config.json` é `"regression"`, não `"single_label_classification"`. Um modelo de regressão devolve **valores dimensionais VAD** diretamente — já na escala certa, tipicamente em [-1, 1] ou [0, 1]. A implementação original aplicava `torch.sigmoid()` a estes valores, assumindo que eram logits de classificação.

A sigmoid espreme qualquer valor para [0.5, 1.0] (para logits positivos) ou [0.0, 0.5] (para logits negativos). Com logits perto de zero (que é o que expected para fala neutra), sigmoid(0) = 0.5 para tudo. Mesmo se a cabeça estivesse corretamente carregada, a sigmoid comprimiria toda a variação para uma janela minúscula à volta de 0.5.

**Como foi corrigido:** Removida a `torch.sigmoid()`. Os outputs do modelo são agora tratados como valores de regressão diretos, clipped para [0, 1] com `np.clip` (proteção contra valores fora do intervalo esperado):

```python
# Antes (errado):
probs = torch.sigmoid(outputs.logits).numpy()[0]

# Depois (correto):
logits = model(**inputs)              # returns tensor directly, no .logits
probs = logits.numpy()[0]
probs = np.clip(probs, 0.0, 1.0)     # safety clip, not sigmoid
```

### Bug 3: Ordem dos outputs trocada

**O problema:** O `config.json` do modelo define claramente a ordem dos outputs:

```json
"id2label": {"0": "arousal", "1": "dominance", "2": "valence"}
```

A implementação original assumia `[valence, arousal, dominance]` (índices 0, 1, 2 respetivamente) e construía o dict de output nessa ordem:

```python
# Antes (errado):
"valence": round(float(probs[0]), 4),    # índice 0 = arousal, NÃO valence
"arousal": round(float(probs[1]), 4),    # índice 1 = dominance, NÃO arousal
"dominance": round(float(probs[2]), 4),  # índice 2 = valence, NÃO dominance
```

Mesmo que o modelo estivesse a funcionar corretamente, **todas as labels estariam trocadas**: o campo `valence` conteria arousal, `arousal` conteria dominance, e `dominance` conteria valence. O gráfico de engagement no relatório mostraria "Prospect Valence" com dados de arousal — enganador e errado.

**Como foi corrigido:** Adicionadas constantes explícitas que mapeiam os índices do modelo aos nomes, e o dict de output é construído na ordem correta:

```python
# Constantes alinhadas com config.json do modelo
AROUSAL_IDX = 0
DOMINANCE_IDX = 1
VALENCE_IDX = 2

# Depois (correto):
"valence":   round(float(probs[VALENCE_IDX]), 4),    # índice 2
"arousal":   round(float(probs[AROUSAL_IDX]), 4),    # índice 0
"dominance": round(float(probs[DOMINANCE_IDX]), 4),  # índice 1
```

### Confirmação após a correção

Depois de corrigir os três bugs, re-corrermos a rúbrica de validação sobre o output real do pipeline (197 segmentos, áudio real de ~12 minutos):

| Check | Antes (broken) | Depois (fixed) | Veredicto |
|---|---|---|---|
| **Spread de valence** | range 0.016, std 0.003 | range **0.781**, std **0.142** | ✅ Sinal claro |
| **Spread de arousal** | range 0.018, std 0.002 | range **0.448**, std **0.068** | ✅ Sinal claro |
| **Spread de dominance** | range 0.009, std 0.002 | range **0.430**, std **0.064** | ✅ Sinal claro |
| **% clustered at 0.5** | 100% in 0.45–0.55 | V=27%, A=58%, D=31% | ✅ Já não é linha plana |
| **Arousal ↔ energy** | r = −0.036 | r = **+0.362** | ✅ A seguir ao sinal real |
| **Valence mais positivo** | (todos ≈ 0.500) | V=**1.000** ("great"), V=**0.219** (cepticismo pricing) | ✅ Discriminação semântica |

### Por que isto é crítico para o projeto

Este projeto existe para provar uma tese à Scale Labs: **que análise multimodal captura insights que análise de transcript-only não consegue** (ver `docs/superpowers/specs/2026-06-30-sales-coach-mvp-design.md:10`). O pitch é a diferença visível entre as duas colunas no relatório side-by-side.

O Step 3 é o primeiro step "multimodal" — é onde a voz começa a contribuir com algo que o texto não diz. Se o voice emotion fosse uma linha plana (todos os segmentos em 0.5), então:

1. **O gráfico de engagement timeline seria uma linha reta** — o critério de sucesso número 4 da spec ("the engagement timeline shows a meaningful signal") falhava, e o relatório parecia vazio.

2. **A LLM (Step 5) não teria nada para analisar** — o prompt multimodal diz "Voice emotion: valence=0.31 (negative), arousal=0.22 (low energy)". Se todos os segmentos fossem 0.5, a LLM não conseguiria distinguir entusiasmo de cepticismo, e o output multimodal seria tão genérico quanto o transcript-only. A killer feature deixa de o ser.

3. **Os momentos críticos não teriam timestamps com sinal emocional** — um "critical moment" baseado em voz requires que a voz mude. Linha plana = nenhum critical moment detetável.

4. **A demo para a Scale Labs falhava o objetivo central** — a entire pitch é "olha o que perdes se só usas transcript". Sem voz, facial, e features acústicas a produzir sinais discriminativos, a demo prova o oposto: que multimodal não acrescenta nada.

Em suma: **estes três bugs não eram um detalho técnico — eram a diferença entre a demo provar a tese ou contradizê-la**. A validação estatística sobre o output real (não sobre mocks) foi o que os apanhou. A lição para o resto do pipeline: cada step que produz dados para o relatório precisa de uma rúbrica de validação que confirma que o output discrimina — não só que os testes passam.