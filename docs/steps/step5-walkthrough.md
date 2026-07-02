# Walkthrough do Step 5 — Análise LLM (transcript-only vs multimodal)

## O que este step faz

O Step 5 é o passo onde os quatro JSONs anteriores (`transcript`, `audio_features`, `voice_emotion`, `face_emotion`) se transformam num **julgamento de vendas**. Carrega os quatro ficheiros, funde-os num bloco compacto por segmento (descartando os timestamps ao nível da palavra — ~200 KB do `transcript.json` que a LLM não precisa), e chama a **Claude API duas vezes** com o *mesmo* system prompt mas *conteúdo diferente*:

1. **transcript-only** — só o texto etiquetado por speaker + timing. Isto é o baseline honesto: o que uma empresa como a Scale Labs já faz hoje (pôr a transcrição numa LLM).
2. **multimodal** — o bloco completo por segmento: texto + features de áudio (pitch, energia, pausas) + emoção na voz (valence/arousal/dominance) + emoção na face.

Os dois outputs têm a mesma estrutura (`engagement_score`, `deal_probability`, `talk_ratio`, `critical_moments[]`, `recommendations[]`) e são gravados em `output/analysis.json` como `{transcript_only, multimodal}`.

**A comparação lado-a-lado entre os dois blocos é a killer feature do projeto** — é o pitch à Scale Labs. O bloco multimodal consegue ver *dissonância cross-modal* (momentos em que as palavras dizem uma coisa e a voz/face dizem outra) que o bloco transcript-only, por construção, **não consegue ver** porque nunca lhe foi dado esse sinal. O relatório (Step 6) põe os dois lado a lado para demonstrar exatamente isso.

## O que cada campo do output significa

`analysis.json` é um objeto com dois blocos paralelos (`transcript_only` e `multimodal`), cada um com estes campos:

| Campo | Tipo | O que significa | Notas |
|-------|------|-----------------|-------|
| `engagement_score` | int 0–100 | Engajamento global do prospect na reunião | Frequentemente mais baixo no multimodal — a voz/face revela desengajamento que o texto esconde |
| `deal_probability` | int 0–100 | Probabilidade estimada de o deal avançar | Mesma lógica — o multimodal é mais cético porque vê negatividade oculta |
| `talk_ratio` | `{rep, prospect}` ints | % de tempo de fala, medido a partir da transcrição | **Idêntico nos dois blocos** — é calculado deterministicamente (`_compute_talk_ratio`), não gerado pela LLM, e injetado em ambos |
| `critical_moments` | array | Momentos-chave, cada um com `timestamp` (HH:MM:SS), `type`, `description`, `coaching` | O multimodal tipicamente tem *mais* momentos porque pode assinalar dissonância que o transcript-only não vê |
| `recommendations` | array de strings | Ações de coaching de topo | Frequentemente sobrepõem-se entre os dois blocos (os *momentos* divergem mais que as *recommendations*) |

**Exemplo (real, do vídeo de demo) — o bloco multimodal:**
```json
{
  "engagement_score": 62,
  "deal_probability": 45,
  "talk_ratio": {"rep": 69, "prospect": 31},
  "critical_moments": [
    {
      "timestamp": "00:12:23",
      "type": "Dissonance – Positive Verbal vs Deeply Sad Facial",
      "description": "Prospect says 'No, it looks great' about the IT budget tool with strong voice valence (0.85) — her highest positive verbal signal of the meeting — but facial emotion is deeply sad",
      "coaching": "This signal indicates the prospect is performing positivity — likely to move the meeting along..."
    }
  ],
  "recommendations": ["..."]
}
```

Repara no `00:12:23`: o prospect diz "it looks great" com a valência na voz mais alta de toda a reunião (0.85), mas a face está profundamente `sad`. A LLM multimodal lê "a prospect está a fazer performance de positividade para despachar a reunião" — algo que o bloco transcript-only, só com o "looks great", regista como sinal positivo. **Isto é o pitch.**

---

## A ferramenta: Anthropic Claude API + tool-use

### Claude (`claude-sonnet-4-6`) — a LLM que produz a análise

A Claude é chamada via `anthropic` Python SDK, duas vezes por execução do Step 5. Mas há uma decisão de design crítica: **não se faz parse de texto livre**. Em vez de pedir à LLM "devolve um JSON com estes campos" e depois fazer parsing da string (frágil — a LLM pode cercar com markdown, esquecer um campo, fechar mal o JSON), usa-se **tool-use forçado**:

```python
tools=[ANALYSIS_TOOL],
tool_choice={"type": "tool", "name": "submit_analysis"},
```

Isto **restringe a LLM** a emitir um bloco `tool_use` cujo `input` é JSON que *tem* de obedecer ao `input_schema` do `ANALYSIS_TOOL`. O output é sempre bem-formado. O schema define `engagement_score`, `deal_probability`, `critical_moments[]` (cada um com `timestamp`/`type`/`description`/`coaching` obrigatórios) e `recommendations[]`. Nota: `talk_ratio` **não** está no schema — é calculado por nós e injetado depois (a LLM é má a fazer aritmética exata).

### Import lazy do `anthropic`

Tal como o pyannote no `diarize.py` e o cv2/deepface no `emotion_face.py`, o `import anthropic` está **dentro** de `_call_claude()`, não no topo do módulo. Consequência: `from pipeline.llm_analysis import run_analysis` é instantâneo e os 63 testes unitários correm em ~0.3s **sem o SDK instalado** — os testes injetam um `anthropic` falso em `sys.modules["anthropic"]` (e limpam-no num `finally`).

---

## Ficheiro 1: `pipeline/llm_analysis.py`

O módulo com a lógica. Está dividido em funções pequenas com responsabilidades claras — funções puras de projeção/formatação (testáveis sem dependências), um *seam* para a API (`_call_claude`), e um orquestrador (`run_analysis`).

### Constantes e system prompt

```python
MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 8192
RATE_LIMIT_WAIT = 10
REQUIRED_FIELDS = ("engagement_score", "deal_probability", "critical_moments", "recommendations")
```

- **`MAX_TOKENS = 8192`** — o orçamento de tokens para a resposta. Era 4096; foi subido para 8192 por causa do bug de truncagem (ver "Bugs encontrados").
- **`REQUIRED_FIELDS`** — os campos que o `input` do tool-use *tem* de ter; `_validate_analysis` verifica-os.

O `SYSTEM_PROMPT` define o papel ("senior B2B sales coach"), as regras (citar timestamps e valores de sinal, sem conselhos genéricos), e — crucialmente — um **guia de interpretação dos sinais** + uma instrução para **superficiar dissonância**:

> "When the words say one thing but the voice/face say another, that dissonance is the most important signal — surface it as a critical moment with the timestamp and the conflicting signal values."

É esta instrução que faz o bloco multimodal produzir os momentos de dissonância que o transcript-only não consegue. O system prompt é o *produto*, mais que os JSONs.

### Funções puras de projeção

`_format_timestamp(seconds) -> "HH:MM:SS"`, `_classify_speakers(transcript) -> {SPEAKER_00: "REP", ...}` (por tempo de fala — o maior = REP), `_compute_talk_ratio(transcript, speaker_map) -> {rep, prospect}` (medido, não LLM), `_face_for_segment(segment, face_emotion) -> nearest face sample` (a face é amostrada a cada 10s sem speaker; cada segmento fica com a amostra mais próxima do midpoint), e `_merge_segments(...)` que funde os três arrays per-segment (197 entradas, alinhados por índice) num array rico, **descartando `words` e `avg_logprob`** e religando speakers a roles.

### `_build_transcript_prompt` vs `_build_multimodal_prompt`

Estas duas funções constroem o conteúdo do user para as duas chamadas. O transcript-only só inclui speaker + texto + timestamp; o multimodal inclui o bloco completo:

```
SEGMENT [00:12:23]
Speaker: PROSPECT
Text: "No, it looks great"
Audio: pitch_mean=... pitch_std=... energy=... speech_rate=... pause_ratio=... zcr=...
Voice emotion: valence=0.85 arousal=... dominance=...
Facial: sad (dominant), scores={...}
```

O contraste entre os dois prompts *é* o contraste entre os dois outputs. O transcript-only envia literalmente só a parte de texto (a "transcrição" que a Scale Labs já tem); o multimodal envia tudo.

### `_call_claude()` — o seam da API

```python
def _call_claude(user_prompt: str, api_key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = _create_message(client, user_prompt)
    except anthropic.RateLimitError:
        time.sleep(RATE_LIMIT_WAIT)
        response = _create_message(client, user_prompt)
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise RuntimeError("Claude response truncated (stop_reason=max_tokens); increase MAX_TOKENS")
    result = _extract_tool_input(response)
    _validate_analysis(result)
    return result
```

1. **`import anthropic`** — lazy, dentro da função.
2. **Retry uma vez** — `RateLimitError` → `sleep(10)` → uma tentativa. A spec exige isto ("retry once after 10 seconds").
3. **Guarda de truncagem** — `stop_reason == "max_tokens"` → raise. (Ver "Bugs".)
4. **`_extract_tool_input`** — tira o `input` do primeiro bloco `tool_use` do `response.content`.
5. **`_validate_analysis`** — verifica que todos os `REQUIRED_FIELDS` estão presentes; se não, raise. Defesa em profundidade: o `tool_choice` forçado garante um *bloco* tool_use, mas não um *completo*.

### `run_analysis()` — o orquestrador

Carrega os 4 JSONs, classifica speakers, calcula `talk_ratio`, funde segmentos, chama `_call_claude` duas vezes (transcript-only + multimodal), e injeta `talk_ratio` em ambos os outputs:

```python
analysis = {
    "transcript_only": {**transcript_llm, "talk_ratio": talk_ratio},
    "multimodal": {**multimodal_llm, "talk_ratio": talk_ratio},
}
```

O `{**llm_output, "talk_ratio": talk_ratio}` põe o `talk_ratio` medido por cima de qualquer coisa (o schema não o pede, mas se a LLM o devolvesse por acaso, o nosso valor é que fica). Grava em `output/analysis.json`.

---

## Ficheiro 2: `pipeline/05_llm_analysis.py`

O CLI — fino, idêntico em estrutura ao `04_emotion_face.py`:

```python
def main():
    missing = [p for p in REQUIRED_INPUTS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: missing input(s): {', '.join(missing)}")
        sys.exit(1)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env (see .env.example).")
        sys.exit(1)
    from llm_analysis import run_analysis
    print("→ Running LLM analysis (transcript-only + multimodal)...")
    run_analysis(TRANSCRIPT_FILE, AUDIO_FEATURES_FILE, VOICE_EMOTION_FILE,
                 FACE_EMOTION_FILE, OUTPUT_FILE, api_key=api_key)
    print(f"✓ Analysis saved to {OUTPUT_FILE}")
```

- **Duas guardas fail-fast** — inputs em falta → exit 1 (mensagem nomeia o ficheiro); `ANTHROPIC_API_KEY` em falta → exit 1. Ambas testadas via subprocess.
- **Bare import** — `from llm_analysis import run_analysis` (não `from pipeline.llm_analysis`), como todos os CLIs numerados.
- **`load_dotenv()`** — carrega `.env` antes de ler a key. Por defeito não faz override de vars de ambiente já definidas (o teste da key em falta explora isto).

---

## Visão Geral do Fluxo

```
output/transcript.json ─┐
output/audio_features.json ─┤
output/voice_emotion.json ─┼─→ run_analysis()
output/face_emotion.json ─┘        │
                                   ├── _classify_speakers()    → SPEAKER_00=REP, _01=PROSPECT, _02=OTHER
                                   ├── _compute_talk_ratio()   → {rep:69, prospect:31}  (medido)
                                   ├── _merge_segments()       → 197 blocos (words dropped, face attached)
                                   │
                                   ├── _call_claude( _build_transcript_prompt(merged) )  → LLM output (só texto)
                                   ├── _call_claude( _build_multimodal_prompt(merged) ) → LLM output (tudo)
                                   │        (tool_choice forçado → JSON garantido; stop_reason + _validate_analysis guardam truncagem)
                                   │
                                   └── injeta talk_ratio em ambos → grava output/analysis.json
                                   ↓
                          {transcript_only: {...}, multimodal: {...}}
```

1. O CLI verifica os 4 inputs + a API key.
2. `run_analysis` carrega os 4 JSONs, projeta-os (drop de `words`/`avg_logprob`, religa speakers a roles, anexa a face mais próxima a cada segmento).
3. Chama a Claude **duas vezes** — transcript-only depois multimodal — com tool-use forçado.
4. Injeta o `talk_ratio` medido em ambos os blocos.
5. Grava `output/analysis.json`.

---

## A killer feature: dissonância multimodal (no vídeo de demo)

No vídeo de demo, os dois blocos produziram:

| | transcript_only | multimodal |
|---|---|---|
| `engagement_score` | 62 | 62 |
| `deal_probability` | 45 | 45 |
| `critical_moments` | 10 | **13** |
| `recommendations` | 6 | 7 |
| `talk_ratio` | {rep:69, prospect:31} | {rep:69, prospect:31} |

O `talk_ratio` é idêntico (medido). O multimodal tem **13 momentos vs 10**, e — mais importante — os 3-4 momentos extra são de **dissonância** que o transcript-only, por construção, não consegue produzir porque nunca lhe foi dado o sinal de voz/face. Exemplos reais do bloco multimodal:

- **`00:12:23` — "looks great" mas face deeply sad.** O prospect diz *"No, it looks great"* com a valência na voz **mais alta de toda a reunião (0.85)**, mas a face está profundamente `sad`. A LLM multimodal: *"the prospect is performing positivity — likely to move the meeting along"*. O transcript-only lê "looks great" como sinal positivo. **A mesma frase, leitura oposta.**
- **`00:10:36` — "Sure" mas face sad a 77.6%.** O prospect diz *"Sure"* (a aprender sobre budget tools) mas a face tem o score `sad` mais forte da reunião (0.776). O "Sure" não reflete o estado real.
- **`00:03:21` — pergunta de medo → face angry spike.** O vendedor pergunta "what if you made a poor decision?" e a face do prospect dispara para `angry` 0.55 (o pico da reunião) antes mesmo de responder. O transcript-only vê a pergunta e a resposta; o multimodal vê a irritação facial imediata que a pergunta provocou.
- **`00:05:09` — "Pretty much just him" com valence 0.23 (o ponto mais baixo).** Linguagem casual/breve, mas a voz afunda para a valência mais baixa da reunião. O transcript-only regista uma resposta curta e neutra; o multimodal vê afeto negativo energizado.

Estes são os momentos que fazem o pitch à Scale Labs: **a coluna transcript-only do relatório vai parecer magra e genérica ao lado da multimodal, que cita timestamps e valores de sinal concretos.** É exatamente o critério de sucesso #2 da spec ("the side-by-side comparison makes transcript-only analysis look visually thin").

---

## Bugs encontrados e corrigidos

Tal como os Steps 3 e 4, o Step 5 **passava em todos os testes unitários mas produzia um output silenciosamente partido**. O bug não foi encontrado por um teste — foi encontrado por um **code reviewer a ler o `output/analysis.json` real** e reparar que o bloco multimodal não tinha a chave `recommendations`. A lição é a mesma dos steps anteriores: *testes unitários com mocks validam a plumbing, não o output real da LLM*.

### O bug: `MAX_TOKENS=4096` truncava o bloco multimodal

**O problema:** Com `max_tokens=4096`, a chamada multimodal produziu 13–15 `critical_moments` muito detalhados (cada um com descrição + coaching) que **consumiram todo o orçamento de tokens antes de `recommendations` ser emitido**. O bloco `tool_use` fechou o array `critical_moments` e o JSON parcial fez parse num dict *válido-mas-incompleto* — faltava `recommendations`. Esse dict incompleto fluiu silenciosamente para `analysis.json`. O bloco multimodal no disco tinha as chaves `['engagement_score', 'deal_probability', 'critical_moments', 'talk_ratio']` — **sem `recommendations`**.

**Por que é que os testes não apanharam isto:**
1. Os testes unitários do `_call_claude` usam um `anthropic` falso que devolve inputs de teste completos — nunca truncam.
2. O teste de integração (live API) *passou* — mas por sorte: numa execução diferente, a LLM não truncou (o comprimento do output varia por chamada). Ou seja, o teste passou numa run lucky enquanto o run do CLI (`python pipeline/05_llm_analysis.py`) truncou. Não-determinismo da LLM.
3. Pior: o subagente que executou a primeira run real *relatou* ter 5 `recommendations` multimodais — mas não as havia no ficheiro. O bug só foi apanhado quando um code reviewer independente leu o `output/analysis.json` real e verificou as chaves.

**O ponto subtal:** `tool_choice={"type": "tool", "name": "submit_analysis"}` garante que existe um *bloco* `tool_use` na resposta — mas **não** garante que esse bloco esteja *completo*. Quando o `max_tokens` se esgota a meio do JSON, o bloco fecha (parsed como dict válido) e o que lá está é o que há. Forçar o tool-use não é defesa contra truncagem.

**Como foi corrigido (três camadas):**
```python
MAX_TOKENS = 8192   # era 4096
```
```python
if getattr(response, "stop_reason", None) == "max_tokens":
    raise RuntimeError("Claude response truncated (stop_reason=max_tokens); increase MAX_TOKENS")
```
```python
def _validate_analysis(result: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in result]
    if missing:
        raise RuntimeError(f"Claude analysis incomplete (missing: {', '.join(missing)}); likely max_tokens truncation")
```

1. **`MAX_TOKENS 4096 → 8192`** — mais orçamento; o bloco multimodal cabe folgado (a chamada real agora produz 13 momentos + 7 recomendações e ainda sobra orçamento).
2. **`stop_reason == "max_tokens"` check** — deteta truncagem explicitamente e falha alto.
3. **`_validate_analysis`** — defesa em profundidade: mesmo sem o `stop_reason` a indicar truncagem, se algum campo obrigatório faltar, falha alto em vez de gravar JSON partido.

### Confirmação após o fix

Re-correr o Step 5 no vídeo de demo, após o fix:

| Check | Antes (broken) | Depois (fixed) | Veredicto |
|---|---|---|---|
| Bloco multimodal tem `recommendations`? | **Não** (0) | **Sim** (7) | ✅ |
| `critical_moments` multimodal | 13 (mas incompleto) | 13 (completo) | ✅ |
| `stop_reason` guard | n/a | implementado + testado | ✅ |
| `_validate_analysis` | n/a | implementado + testado | ✅ |
| Teste de integração live | passou por sorte | passa com as novas asserções (multimodal cita valence/arousal/face; transcript-only não) | ✅ |

Foram adicionados testes que alimentam um `tool_use` *incompleto* (sem `recommendations`) ao `_call_claude` e verificam que ele **levanta** em vez de gravar JSON partido — pelo que o não-determinismo que escondeu o bug não pode recurring.

### Por que isto é crítico para o projeto

O `analysis.json` é o artefacto que o Step 6 (relatório) consome e que a Scale Labs vai ver. Se o bloco multimodal tivesse sido shipado sem `recommendations`:

1. **A killer feature ficava partida** — a coluna multimodal do relatório teria momentos mas não recomendações, enquanto a transcript-only teria ambas. A comparação lado-a-lado pareceria partida, não "magra vs rica".
2. **Falha silenciosa** — o JSON era *válido* (parsava), pelo que o Step 6 ia renderizar o bloco multimodal como se estivesse completo. Nenhum erro, nenhum crash — só um output subtilmente errado que descredibilizaria a demo.

A lição, igual à dos Steps 3 e 4: **cada step que produz um artefacto para o relatório precisa de validação sobre o output real**, não só sobre testes com mocks. Aqui, os testes com mocks confirmavam que o `_call_claude` extraía o `tool_use` input e o gravava — tudo correto para a API que tínhamos definido. Mas a API tinha um pressuposto errado: que `tool_choice` forçado = output completo. Só um reviewer a ler o `analysis.json` real e verificar as chaves é que apanhou isto.
