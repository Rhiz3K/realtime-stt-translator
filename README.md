# Czech Live Translator

Jednoúčelová webová aplikace pro živý překlad české řeči do angličtiny a
ruštiny. Nemá volbu enginu, modelu, jazyků ani velikosti audio bloků. Mikrofonní
audio jde přes chráněný FastAPI WebSocket přímo do Google Gemini API; API klíč
zůstává pouze na serveru.

```text
mikrofon (PCM16, mono, 16 kHz)
  → gemini-3.5-transcribe-live (cs-CZ, SMART)
  → český interim/final přepis
  → jeden Flash-Lite požadavek
  → {"en": "…", "ru": "…"}
```

## Pevné chování

- Google Live Transcribe používá `gemini-3.5-transcribe-live`, jazyk `cs-CZ` a
  režim `SMART`.
- Pro časté průběžné překlady se používá levný `gemini-2.5-flash-lite`.
- Každý finální přepis se ihned překládá přes `gemini-3.5-flash-lite`.
- EN a RU vzniknou v jednom structured-output požadavku a do prohlížeče dorazí
  atomicky v jednom JSON objektu.
- Rychle se měnící interim hypotézy se slučují na jeden průběžný požadavek a
  jednu nejnovější čekající hodnotu; finální překlady mají vždy přednost.
- Server přijímá audio nejvýše rychlostí reálného času a současně povolí čtyři
  placené audio relace; tím omezuje škodu způsobenou vadným klientem.
- Audio se neukládá a STT probíhá výhradně v Google API.

## Cena a limit relace

Live Transcribe stojí orientačně **$0.009 za minutu audia**, tedy přibližně
**$0.09 za plnou desetiminutovou relaci**. K tomu se přičte textový překlad:

| Model | Vstup / 1M tokenů | Výstup / 1M tokenů |
|---|---:|---:|
| `gemini-2.5-flash-lite` (interim) | $0.10 | $0.40 |
| `gemini-3.5-flash-lite` (final) | $0.30 | $2.50 |

Textová část bývá proti ceně audia malá, ale její přesná cena závisí na délce
řeči a počtu změn interim přepisu. Aktuální ceny vždy ověřte v
[oficiálním ceníku Gemini API](https://ai.google.dev/gemini-api/docs/pricing).

Google omezuje jednu Live Transcribe session na **10 minut**. Server proto relaci
řízeně ukončí po 9 minutách a 45 sekundách, nechá doběhnout poslední final a v UI
umožní okamžitě spustit novou relaci.

## Spuštění

Požadavky: Python 3.12 a Google Gemini API klíč.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

V `.env` nastavte alespoň:

```dotenv
GEMINI_API_KEY=...
APP_PASSWORD=...
AUTH_SECRET=...
```

Potom spusťte:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Aplikace je na `http://localhost:8000`. Mimo localhost vyžaduje přístup k
mikrofonu HTTPS.

## Konfigurace

STT a překlad nemají žádné konfigurovatelné volby. Podporované proměnné jsou
pouze přístupové a provozní:

| Proměnná | Výchozí hodnota | Význam |
|---|---|---|
| `GEMINI_API_KEY` | — | Povinný serverový Google Gemini API klíč. |
| `APP_PASSWORD` | — | Heslo aplikace, pokud je přihlášení zapnuté. |
| `AUTH_ENABLED` | `true` | `false` použijte jen za důvěryhodnou autentizační proxy. |
| `AUTH_SECRET` | `APP_PASSWORD` | HMAC podpis cookies; v produkci použijte samostatný náhodný secret. |
| `AUTH_COOKIE_NAME` | `translator_auth` | Název autentizační cookie. |
| `AUTH_TOKEN_TTL_SECONDS` | `43200` | Platnost přihlášení v sekundách. |
| `AUTH_COOKIE_SECURE` | podle schématu | V produkci za HTTPS nastavte `true`. |
| `ALLOWED_ORIGINS` | stejný host | Volitelný seznam přesných WebSocket originů oddělený čárkou. |
| `PORT` | `8000` | Port použitý Docker startovacím příkazem. |
| `FORWARDED_ALLOW_IPS` | Uvicorn default | Důvěryhodné proxy pro `X-Forwarded-*`. |

## WebSocket protokol

Jediný audio endpoint je `WS /ws/audio`. Prohlížeč posílá binární raw PCM16 LE,
mono, 16 kHz v interně pevných 100ms blocích. Při zastavení pošle pouze:

```json
{"type":"stop"}
```

Server odpovídá stavem a překlady:

```json
{"type":"ready"}
{"type":"interim","en":"Hello","ru":"Здравствуйте"}
{"type":"final","en":"Hello.","ru":"Здравствуйте."}
{"type":"ended"}
```

Chyby mají stabilní kód bez detailů nebo API klíče:

```json
{"type":"error","code":"google_unavailable","recoverable":true}
```

## Docker

```bash
docker build -t czech-live-translator .
docker run --rm --env-file .env -p 8000:8000 czech-live-translator
```

Kontejner běží pod neprivilegovaným uživatelem. `GET /health/live` ověřuje běh
procesu a `GET /health/ready` vrací 200 pouze s použitelnou Google a auth
konfigurací. Docker `HEALTHCHECK` používá `/health/ready`.

Za reverzní proxy povolte WebSocket upgrade, používejte HTTPS a správně nastavte
`FORWARDED_ALLOW_IPS` i `ALLOWED_ORIGINS`.

## Kontroly

```bash
python -m compileall app tests
pytest
pytest --cov=app --cov-report=term-missing
pip-audit -r requirements.txt
docker build --check .
docker build .
```

CI provádí audit runtime závislostí, pytest, validaci Dockerfile a sestavení
produkčního obrazu.
