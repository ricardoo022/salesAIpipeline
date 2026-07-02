# Walkthrough do Step 4 — Emoção na Face

## O que este step faz

O Step 4 analisa o **rosto** das pessoas no vídeo da reunião. Ao contrário do Step 3 (que analisa a emoção na *voz* segmento a segmento), o Step 4 não depende da diarização nem da transcrição — funciona diretamente sobre o ficheiro de vídeo `input/meeting.mp4`. A cada **10 segundos** tira uma *frame* do vídeo (uma imagem estática), corrê o **DeepFace** sobre essa frame para classificar a expressão facial em 7 emoções discretas (`angry, disgust, fear, happy, sad, surprise, neutral`), e regista a emoção dominante mais o *score* de cada uma. Frames onde o DeepFace não encontra nenhuma cara são **ignoradas silenciosamente** (não há crash — a spec exige isto explicitamente). O output é `output/face_emotion.json` — um array com um objeto por frame analisada, cada um com `{timestamp, dominant_emotion, scores}`.

A granularidade é mais grosseira que a do Step 3 (uma leitura a cada 10s vs. uma por turno de fala), mas isto é propositado: a expressão facial muda devagar, e o DeepFace é pesado em CPU. 10 segundos dá um equilíbrio entre resolução temporal e tempo de execução (~90 frames para um vídeo de 15 minutos, ~10 minutos em CPU).

## O que cada campo do output significa

Cada frame analisada no `output/face_emotion.json` tem estes campos:

| Campo | O que significa | Por que é importante em vendas | Como foi calculado |
|-------|----------------|-------------------------------|--------------------|
| `timestamp` | Segundo do vídeo em que a frame foi amostrada (ex: `100.0`) | Permite alinhar a expressão facial com o que foi dito nesse momento (cruzando com o `transcript.json`) e com a emoção na voz do Step 3 | De 10 em 10 segundos (`SAMPLE_INTERVAL = 10`); arredondado a 2 casas |
| `dominant_emotion` | A emoção com `score` mais alto nessa frame (ex: `"neutral"`) | Leitura instantânea do estado facial — para o relatório e para a LLM | `argmax(scores)` — o DeepFace já devolve este campo |
| `scores` | Dicionário com as 7 emoções e um valor 0–1 para cada (a probabilidade de cada uma, normalizada a partir das percentagens 0–100 do DeepFace) | Granularidade para a LLM (Step 5): em vez de só "neutral", a LLM vê `neutral=0.76, happy=0.21, sad=0.03` e pode inferir "cliente neutro mas com traço de preocupação" | DeepFace emotion model → percentagens 0–100 → dividir por 100 → arredondar a 4 casas |

> **Porquê 7 emoções discretas e não VAD como o Step 3?** Os dois steps usam representações diferentes de propósito. O Step 3 (voz) usa VAD contínuo (valence/arousal/dominance) porque a voz varia de forma fluida e dimensional. O Step 4 (face) usa categorias discretas porque a expressão facial é mais categórica (um sorriso é `happy`, uma sobrancelhal franzida é `angry`). No Step 5, a LLM combina as duas representações — a dimensional da voz com a categórica da face — para uma leitura multimodal mais rica.

**Exemplo de uma frame do output (real, t=100s do vídeo de demo):**
```json
{
  "timestamp": 100.0,
  "dominant_emotion": "neutral",
  "scores": {
    "angry": 0.0035,
    "disgust": 0.0,
    "fear": 0.0037,
    "happy": 0.205,
    "sad": 0.0279,
    "surprise": 0.0,
    "neutral": 0.7598
  }
}
```

A frame é maioritariamente `neutral` (0.76) mas com um traço de `happy` (0.21) — um cliente atento mas relaxado, não entediado nem entusiasmado. A LLM (Step 5) usa esta granularidade em vez de só ler "neutral".

---

## A ferramenta: DeepFace + OpenCV

Antes do código, vale a pena perceber as duas bibliotecas envolvidas.

### OpenCV (`opencv-python`) — amostragem de frames

O OpenCV é usado **só para tirar frames do vídeo**, não para detetar caras. A função `_iter_frames` abre o vídeo com `cv2.VideoCapture`, lê os metadados (fps, número total de frames), e procura a frame mais próxima de cada marca de 10 segundos usando `CAP_PROP_POS_FRAMES` (seek por índice de frame). Para um vídeo de 15 minutos a 25 fps (22 336 frames), amostra ~90 frames em vez de processar as 22 336 todas — muito mais rápido.

### DeepFace — classificação de emoção facial

DeepFace é um framework de reconhecimento facial que junta vários modelos pré-treinados. Para o Step 4 usamos duas funcionalidades:

- **Deteção de cara** (`detector_backend`) — encontra onde está a cara na frame antes de a classificar. O DeepFace suporta vários backends: `opencv` (haarcascade), `retinaface`, `mtcnn`, `ssd`, `yolov8`, etc. **Esta escolha foi o primeiro bug encontrado (ver secção "Bugs")** — o backend default (`opencv`) não funciona com `opencv-python` 5.x, por isso usamos `retinaface`.
- **Classificação de emoção** (`actions=["emotion"]`) — um modelo pré-treinado (o `facial_expression_model_weights.h5`) que produz 7 scores de emoção. O DeepFace devolve estes scores como **percentagens 0–100** (não 0–1), o que foi o segundo bug (ver "Bugs").

O DeepFace faz download automático dos pesos dos modelos para `~/.deepface/weights/` na primeira execução (retinaface ~1 MB, emotion model ~64 MB) e reutiliza-os nas seguintes.

> **Por que `enforce_detection=True`?** Quando o DeepFace não encontra nenhuma cara numa frame, com `enforce_detection=True` ele **lança um `ValueError`** em vez de devolver um resultado falso. O `_analyze_frame` apanha esse `ValueError` e devolve `None`, sinalizando "saltar esta frame". Isto garante que o output só contém leituras de caras reais — nunca emoções inventadas sobre fundos sem cara.

---

## Ficheiro 1: `pipeline/emotion_face.py`

O módulo com a lógica principal. Está dividido em 4 funções pequenas, cada com uma responsabilidade clara (uma por linha do fluxo: amostrar → analisar → normalizar → orquestrar). As dependências pesadas (`cv2`, `deepface`) são **lazy-imported** dentro das funções que as usam — o módulo carrega instantaneamente sem torch/TensorFlow, o que permite correr os testes unitários sem esses pesos.

### Imports e constantes

```python
import os

SAMPLE_INTERVAL = 10
DETECTOR_BACKEND = "retinaface"
```

- **`os`** — para verificar se o vídeo existe (`os.path.exists`).
- **`SAMPLE_INTERVAL = 10`** — intervalo de amostragem em segundos. Define a resolução temporal do Step 4.
- **`DETECTOR_BACKEND = "retinaface"`** — o backend de deteção de cara do DeepFace. Há um comentário extenso a explicar por que não é o default (`opencv`) — ver "Bugs encontrados e corrigidos".

> **Nota sobre lazy imports:** Repara que não há `import cv2` nem `import deepface` no topo do ficheiro. Eles estão dentro de `_iter_frames` e `_analyze_frame` respetivamente. Isto espelha o padrão do `pipeline/diarize.py` com o pyannote: as dependências pesadas só são carregadas quando a função que as precisa é realmente chamada. Consequência prática: `from pipeline.emotion_face import extract_face_emotion` é instantâneo; os testes unitários (que fazem patch dessas funções) correm em ~0.6s sem nunca carregar TensorFlow.

### Função `_shape_emotion_result()`

```python
def _shape_emotion_result(raw: dict) -> dict:
    """Normalize a DeepFace emotion analysis result into our output shape."""
    if isinstance(raw, list):
        raw = raw[0]
    emotions = raw["emotion"]
    dominant = raw["dominant_emotion"]
    scores = {k: round(float(v) / 100, 4) for k, v in emotions.items()}
    return {"dominant_emotion": dominant, "scores": scores}
```

Função pura (sem I/O, sem dependências) que normaliza o output bruto do DeepFace no formato do nosso JSON. Faz três coisas:

1. **Desempacotar a lista** — o `DeepFace.analyze` pode devolver um dict (uma cara) ou uma lista de dicts (várias caras). Se for lista, ficamos com a primeira cara (`raw[0]`). Para uma reunião de vendas com duas pessoas, há quase sempre uma cara dominante por frame.
2. **Extrair os campos** — `raw["emotion"]` é o dicionário de 7 emoções → scores; `raw["dominant_emotion"]` é a emoção com score mais alto (o DeepFace já calcula isto).
3. **Normalizar 0–100 → 0–1** — `round(float(v) / 100, 4)`. O DeepFace devolve percentagens (ex: `happy: 99.51`); a spec quer 0–1 (ex: `0.9951`). A divisão por 100 é a correção do segundo bug (ver "Bugs"). O `round(..., 4)` segue a convenção do resto do pipeline (Step 3 arredonda VAD a 4 casas).

Esta função é a única que é testada de forma pura (sem mocks de dependências) — é lógica de transformação simples, e os testes cobrem extração, arredondamento, desempacotamento de lista, e normalização.

### Função `_analyze_frame()`

```python
def _analyze_frame(frame) -> dict | None:
    """Run DeepFace emotion analysis on a single frame."""
    from deepface import DeepFace
    try:
        raw = DeepFace.analyze(
            frame, actions=["emotion"], enforce_detection=True, detector_backend=DETECTOR_BACKEND
        )
    except ValueError:
        return None
    return _shape_emotion_result(raw)
```

O *seam* do DeepFace — a função que isola a chamada à biblioteca pesada, para o resto do módulo poder ser testado sem a carregar. Recebe uma frame (array numpy da imagem) e devolve o dict normalizado `{dominant_emotion, scores}` ou `None`.

1. **`from deepface import DeepFace`** — import lazy, dentro da função. Só aqui é que o TensorFlow é carregado.
2. **`DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True, detector_backend=DETECTOR_BACKEND)`** — a chamada principal. `actions=["emotion"]` diz ao DeepFace para só classificar emoção (não idade/género/raça). `enforce_detection=True` faz o DeepFace lançar `ValueError` se não houver cara. `detector_backend=DETECTOR_BACKEND` usa retinaface (não o default quebrado).
3. **`except ValueError: return None`** — apanha o "no face" e devolve `None`. A função chamadora (`extract_face_emotion`) interpreta `None` como "saltar esta frame". (Ver "Bugs" para uma nuance sobre este catch amplo.)
4. **`return _shape_emotion_result(raw)`** — em sucesso, normaliza o output bruto.

O retorno `dict | None` é o contrato: `None` = sem cara, saltar; dict = emoção classificada.

### Função `_iter_frames()`

```python
def _iter_frames(video_path: str, interval: int = SAMPLE_INTERVAL):
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or total_frames <= 0:
            return
        duration = total_frames / fps
        timestamp = 0.0
        while timestamp < duration:
            frame_idx = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                yield timestamp, frame
            timestamp += interval
    finally:
        cap.release()
```

Um *gerador* que produz pares `(timestamp, frame)` a cada `interval` segundos. É o *seam* do OpenCV.

1. **`import cv2`** — lazy, dentro da função.
2. **`cap = cv2.VideoCapture(video_path)`** — abre o vídeo.
3. **`fps`, `total_frames`** — lê os metadados. Se o vídeo não for legível (fps=0 ou sem frames), o `if not fps or total_frames <= 0: return` termina o gerador cedo (devolve nada — o chamador recebe uma lista vazia, sem crash).
4. **`duration = total_frames / fps`** — duração total em segundos.
5. **`while timestamp < duration:`** — itera de 0 em `interval` (10s) até ao fim do vídeo.
6. **`frame_idx = int(timestamp * fps)`** — converte o tempo (segundos) no índice da frame correspondente (ex: t=10s, fps=25 → frame 250).
7. **`cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)` + `cap.read()`** — salta para essa frame e lê-a. O `ret` é `True` se a leitura teve sucesso.
8. **`yield timestamp, frame`** — produz o par. O `yield` (em vez de `return`) faz disto um gerador: a frame é produzida e processada uma de cada vez, sem carregar as 90 frames todas em memória.
9. **`finally: cap.release()`** — liberta o `VideoCapture`. O `finally` corre sempre: fim normal do loop, o `return` cedo, ou o gerador ser abandonado pelo chamador (GC/close → `GeneratorExit` → `finally`).

### Função `extract_face_emotion()`

```python
def extract_face_emotion(video_path: str, interval: int = SAMPLE_INTERVAL) -> list[dict]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    results = []
    for timestamp, frame in _iter_frames(video_path, interval):
        analysis = _analyze_frame(frame)
        if analysis is None:
            print(f"  ⚠ no face detected at {timestamp:.1f}s, skipping")
            continue
        results.append({
            "timestamp": round(float(timestamp), 2),
            "dominant_emotion": analysis["dominant_emotion"],
            "scores": analysis["scores"],
        })
    return results
```

A função principal, análoga à `extract_voice_emotion()` do Step 3. Orquestra as três funções auxiliares.

1. **Guarda fail-fast** — `if not os.path.exists(video_path): raise FileNotFoundError(...)`. Verifica o vídeo antes de tocar em cv2/deepface. (O teste `test_raises_when_video_missing` verifica isto; o CLI traduz este erro num `exit(1)` com mensagem.)
2. **Loop sobre as frames** — `for timestamp, frame in _iter_frames(...)`: consome o gerador.
3. **`analysis = _analyze_frame(frame)`** — classifica a frame.
4. **Skip sem cara** — `if analysis is None: print(...); continue`. Não há crash; a frame é saltada e um `⚠` é impresso para observabilidade. A spec exige este comportamento ("If no face detected: skip frame, log warning, continue — no crash").
5. **Constrói o registo** — `{timestamp (round 2), dominant_emotion, scores}`. O timestamp é arredondado a 2 casas (a granularidade é de segundos, 4 casas seria ruído).

---

## Ficheiro 2: `pipeline/04_emotion_face.py`

O orquestrador — o script que o `run.py` chama.

```python
import json
import os
import sys

VIDEO_FILE = "input/meeting.mp4"
OUTPUT_FILE = "output/face_emotion.json"


def main():
    if not os.path.exists(VIDEO_FILE):
        print(f"ERROR: {VIDEO_FILE} not found. Place the meeting video at input/meeting.mp4.")
        sys.exit(1)

    from emotion_face import extract_face_emotion, SAMPLE_INTERVAL

    print(f"→ Extracting facial emotion (one frame every {SAMPLE_INTERVAL}s)...")
    emotions = extract_face_emotion(VIDEO_FILE, interval=SAMPLE_INTERVAL)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(emotions, f, indent=2)

    print(f"✓ Facial emotion saved to {OUTPUT_FILE} ({len(emotions)} frames)")


if __name__ == "__main__":
    main()
```

Estrutura idêntica ao `03_emotion_voice.py`:

- **Constantes** — `VIDEO_FILE` (input direto do utilizador, não um intermediário) e `OUTPUT_FILE`.
- **Guarda** — se `input/meeting.mp4` não existe, imprime mensagem clara e sai com código 1. Isto acontece **antes** do `from emotion_face import ...` (que está dentro de `main()`), por isso a validação é instantânea — não espera pelo carregamento do TensorFlow. O mesmo padrão de import deferido do Step 3 com o pyannote.
- **Bare import** — `from emotion_face import ...` (não `from pipeline.emotion_face`), porque quando o script corre diretamente o Python adiciona a diretoria `pipeline/` ao `sys.path`. Espelha o `from emotion_voice import ...` do Step 3.
- **Output** — `json.dump(emotions, f, indent=2)` com `os.makedirs("output", exist_ok=True)`. Convenção de todos os steps.

> **Idempotência:** O `run.py` (linhas 25–30) tem o Step 4 registado com `output/face_emotion.json` como ficheiro de output. Na linha 59, `if os.path.exists(output): ... continue` — se o JSON já existe, o `run.py` salta o Step 4. Para re-correr: `rm output/face_emotion.json && python run.py`.

---

## Visão Geral do Fluxo

```
input/meeting.mp4 ──────→ pipeline/emotion_face.py:extract_face_emotion()
                              │
                              ├── _iter_frames()  (cv2, lazy)  → (timestamp, frame) a cada 10s
                              │
                              └── for each frame:
                                    ├── _analyze_frame()  (deepface, lazy)
                                    │     ├── DeepFace.analyze(enforce_detection=True, retinaface)
                                    │     ├── if ValueError (no face) → return None → skip
                                    │     └── _shape_emotion_result() → {dominant_emotion, scores 0-1}
                                    │
                                    └── {timestamp, dominant_emotion, scores}
                              ↓
                      output/face_emotion.json
```

1. O `04_emotion_face.py` verifica se `input/meeting.mp4` existe.
2. Importa `extract_face_emotion` (deferido — só aqui carrega cv2 + deepface + TensorFlow).
3. Chama `extract_face_emotion(VIDEO_FILE, interval=10)`.
4. A função:
   - Verifica o vídeo (fail-fast).
   - `_iter_frames` amostra uma frame a cada 10s via OpenCV (seek por índice).
   - Para cada frame: `_analyze_frame` corre o DeepFace (retinaface deteta a cara, emotion model classifica).
   - Se não houver cara: salta (não crasha).
   - Se houver cara: normaliza os scores 0–100 → 0–1.
   - Constrói o registo `{timestamp, dominant_emotion, scores}`.
5. O resultado é gravado em `output/face_emotion.json`.

O JSON final tem este aspeto (90 frames para o vídeo de demo de ~15 min):
```json
[
  {
    "timestamp": 0.0,
    "dominant_emotion": "happy",
    "scores": {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 0.9951, "sad": 0.0, "surprise": 0.0, "neutral": 0.0049}
  },
  {
    "timestamp": 100.0,
    "dominant_emotion": "neutral",
    "scores": {"angry": 0.0035, "disgust": 0.0, "fear": 0.0037, "happy": 0.205, "sad": 0.0279, "surprise": 0.0, "neutral": 0.7598}
  },
  ...
]
```

---

## Interpretação em contexto de vendas

A distribuição real do vídeo de demo (90 frames) foi:

| Emoção | Frames | % |
|--------|-------|----|
| neutral | 60 | 67% |
| sad | 21 | 23% |
| happy | 5 | 6% |
| angry | 4 | 4% |

Isto é um sinal **significativo, não uma linha plana** — cumpre o critério de sucesso #4 da spec. O dominante `neutral` com picos de `sad` é plausível para uma reunião de vendas B2B: o cliente está atento (neutral) mas com momentos de preocupação/cepticismo (sad), com alguns sorrisos (happy) e pontos de tensão (angry).

O relatório final (Step 6) e a LLM (Step 5) usam estes dados para:

1. **Análise comparativa multimodal** — o pitch central do projeto. A LLM recebe, por momento, a emoção na voz (VAD, Step 3) E a emoção na face (7 categorias, Step 4). Exemplo de insight que só a multimodal capta:
   - Cliente diz "parece interessante" (texto positivo) com valence baixo na voz e expressão facial `neutral`/`sad` → a LLM multimodal deteta "não está convencido", enquanto a transcript-only regista "interessante" como positivo.
   - `happy` a pico na face quando o vendedor fala de pricing → interesse genuíno (cruzado com arousal alto na voz).

2. **Sinal temporal** — os 90 pontos ao longo de 15 minutos mostram a *trajetória* emocional do cliente, não só instantes. Uma descida de `happy`/`neutral` para `sad`/`angry` ao longo da reunião = perda de interesse.

**Padrões a observar:**
- **Muito `neutral` + pouco `happy`** = cliente desengajado, a ouvir por educação.
- **Pico de `angry` num momento de pricing** = objeção forte (cruzar com dominance alto na voz).
- **`happy` sincronizado com momentos-chave** (demo, benefícios) = a ir bem.
- **`sad` sustentado** = cepticismo/worry, não raiva — requer pergunta aberta do vendedor, não réplica.

---

## Bugs encontrados e corrigidos

Tal como o Step 3, o Step 4 **passava em todos os testes unitários mas produzia output inútil** quando corrêmos no vídeo real. Os testes validavam a plumbing (a função chama o DeepFace com os argumentos certos, apanha o `ValueError`, normaliza com `round`), mas não validavam o comportamento real da biblioteca. Os dois bugs só foram descobertos ao **correr o Step 4 no vídeo de demo e inspecionar o output real** — exatamente a mesma lição do Step 3: *testes unitários com mocks validam a plumbing, não o modelo/biblioteca*.

### Como os bugs foram descobertos

Depois de os 15 testes passarem, corrêmos `python pipeline/04_emotion_face.py` no `input/meeting.mp4` (vídeo de ~15 min, 90 frames esperadas). O resultado:

```
  ⚠ no face detected at 830.0s, skipping
  ⚠ no face detected at 840.0s, skipping
  ...
  ⚠ no face detected at 890.0s, skipping
✓ Facial emotion saved to output/face_emotion.json (0 frames)
```

**0 frames.** Todas as 90 frames reportadas como "no face" e saltadas. O comportamento de skip funcionava (não houve crash), mas o output estava vazio — o exato failure mode que invalida o step. Aplicámos o `systematic-debugging` skill (Phase 1: root cause investigation antes de qualquer fix).

### Bug 1: Detector backend default quebrado → 0 frames

**O problema:** O `_analyze_frame` chamava `DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True)` sem especificar `detector_backend`, pelo que o DeepFace usava o backend **default: `opencv`** (haarcascade). Esse backend precisa do ficheiro `haarcascade_frontalface_default.xml` em `cv2/data/`. Mas o **`opencv-python` 5.x envia essa diretoria vazia** (só `__init__.py`) — os XMLs da haarcascade foram removidos do pacote. Consequência: o DeepFace lançava um `ValueError` ("Confirm that opencv is installed... Expected path .../haarcascade_frontalface_default.xml violated") em **cada frame**, e o nosso `except ValueError: return None` engolia-o silenciosamente como "no face". Daí 0 frames.

O diagnóstico passou por duas hipóteses:
1. *As frames amostradas estavam corrompidas/blank* (seek por índice em H.264 é por vezes unreliable). **Desconfirmada** — um script de diagnóstico mostrou frames válidas (mean ~60, max 255, not blank).
2. *O DeepFace falhava por razões de config, não de conteúdo*. **Confirmada** — a mensagem de erro do DeepFace apontava para o haarcascade em falta.

**Como foi corrigido:** Adicionada a constante `DETECTOR_BACKEND = "retinaface"` e passada ao `DeepFace.analyze(..., detector_backend=DETECTOR_BACKEND)`. O `retinaface` é uma dependência do próprio `deepface` (sempre instalado quando o deepface está) e traz os seus pesos próprios (auto-download para `~/.deepface/weights/`), pelo que não depende dos XMLs em falta do `opencv-python`. Verificado: a mesma frame t=10s que antes dava "no face" passou a devolver `dominant=neutral` com retinaface.

**Lição:** O `except ValueError` amplo mascarou um erro de configuração como "no face" — o reviewer de código tinha assinalado isto como *minor* ("bare except ValueError is broad") e foi inicialmente rejeitado como não-crítico. A realidade provou que era mais sério: um catch amplo pode transformar um erro de config numa falha silenciosa. A correção do root cause (backend certo) fez o mask deixar de matter para este bug, mas é uma armadilha a lembrar.

### Bug 2: Scores em 0–100 em vez de 0–1

**O problema:** Depois de o Bug 1 ser corrigido, o Step 4 produziu 90 frames — mas os `scores` estavam em **0–100**, não em 0–1 como a spec define (`docs/.../2026-06-30-sales-coach-mvp-design.md:135` mostra `{"happy": 0.03, ...}`). O DeepFace 0.0.100 devolve as emoções como **percentagens** (ex: `happy: 99.51`), não probabilidades 0–1. O `_shape_emotion_result` fazia `round(float(v), 4)` — arredondava mas não normalizava. Output real: `{"happy": 99.5128, "neutral": 0.4872}`.

Isto quebraria o Step 5 (o prompt da LLM foi escrito contra 0–1: `"Facial: neutral (0.71 confidence)"`) e o Step 6 (o Chart.js pode ter eixos 0–1 hardcoded), e era inconsistente com o Step 3 (VAD em 0–1).

**Como foi corrigido:** Uma linha em `_shape_emotion_result`:
```python
# Antes (errado):
scores = {k: round(float(v), 4) for k, v in emotions.items()}        # 99.5128

# Depois (correto):
scores = {k: round(float(v) / 100, 4) for k, v in emotions.items()}  # 0.9951
```
Verificado no output real: `min=0.0000, max=0.9994`, todos os scores em [0, 1], `dominant_emotion == argmax(scores)` em todas as 90 frames.

### Confirmação após as correções

Re-correr o Step 4 no vídeo de demo, após ambos os fixes:

| Check | Antes (broken) | Depois (fixed) | Veredicto |
|---|---|---|---|
| Frames produzidas | 0 (todas saltadas) | **90** | ✅ |
| Range dos scores | 0–100 (ex: 99.51) | **0–1** (max 0.9994) | ✅ Match à spec |
| `dominant_emotion == argmax(scores)` | n/a (sem frames) | **0 mismatches em 90** | ✅ Consistente |
| Timestamps | n/a | 0.0 → 890.0, gaps de 10.0 | ✅ Amostragem correta |
| Distribuição | n/a | neutral=60, sad=21, happy=5, angry=4 | ✅ Sinal, não flat line |

### Por que isto é crítico para o projeto

O Step 4 é a segunda metade do argumento multimodal (a face). Se produzisse 0 frames ou scores na escala errada:

1. **O pitch multimodal ficava coxo** — a comparação side-by-side do relatório (transcript-only vs multimodal) precisa da coluna facial com dados reais. Sem face, o "multimodal" é só voz + áudio, e a tese de que vale a pena não se demonstra totalmente.
2. **A LLM (Step 5) seria alimentada com lixo** — scores 0–100 num prompt escrito para 0–1 levaria a LLM a interpretações erradas ("happy=99" lido como 99% ou como 0.99 consoante o contexto), descredibilizando o output.
3. **0 frames = relatório vazio** — o critério de sucesso #3 da spec ("at least one critical moment with a timestamp that matches something real") precisa de sinal facial para ancorar os momentos críticos.

A lição, igual à do Step 3: **cada step que produz dados para o relatório precisa de uma validação sobre o output real**, não só sobre os testes. Os testes com mocks confirmaram que a função chamava o DeepFace, apanhava o `ValueError`, e construía o dict — tudo correto para a API que tínhamos definido. Mas a API tinha dois pressupostos errados sobre a biblioteca real (o backend default funciona; os scores vêm em 0–1). Só correr no vídeo real e inspecionar os valores apanhou isto.
