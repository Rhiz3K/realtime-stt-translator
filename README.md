# Czech Live Translator

Jednoúčelová webová aplikace pro živý překlad české řeči do angličtiny a
ruštiny. Nemá volbu enginu, modelu, jazyků ani velikosti audio bloků. Hlavní STT
je opět **Web Speech** (`cs-CZ`, continuous, interim). Server dostává jen české
přepisy a překládá je přes Google. Dosavadní Google audio je **automatická
záloha** při nedostupnosti nebo selhání Web Speech. API klíč zůstává na serveru.

```text
mikrofon → Web Speech → /ws/transcript (český interim/final)
  → jeden Flash-Lite požadavek
  → {"en": "…", "ru": "…"}

automatická záloha:
mikrofon → /ws/audio (PCM16, mono, 16 kHz)
  → gemini-3.5-transcribe-live (cs-CZ, SMART) → stejný překlad EN + RU
```

## Pevné chování

- Web Speech posílá průběžné hypotézy i finaly rovnou do překladače. Při této
  cestě se placené Google Live STT **vůbec neotevírá**. PCM se používá jen
  místně pro statistiky a kontrolu, zda rozpoznávání nezamrzlo.
- Google záloha používá `gemini-3.5-transcribe-live`, jazyk `cs-CZ` a `SMART`.
- Pro časté průběžné překlady se používá `gemini-3.1-flash-lite` s minimálním
  thinking. Původní 2.5 v nasazeném projektu vracel 404; dostupnost 3.1 byla
  ověřena skutečným strukturovaným požadavkem.
- Každý finální přepis se ihned překládá přes `gemini-3.5-flash-lite`.
- EN a RU vzniknou v jednom structured-output požadavku a do prohlížeče dorazí
  atomicky v jednom JSON objektu.
- Rychle se měnící interim hypotézy se slučují na jeden průběžný požadavek a
  jednu nejnovější čekající hodnotu; finální překlady mají vždy přednost.
- Server přijímá audio nejvýše rychlostí reálného času a současně povolí čtyři
  placené relace dohromady na obou cestách **na proces**. Textový vstup má limit
  12 000 znaků, 20 zpráv/s s rezervou 40 zpráv a omezenou frontu finalů.
  Také limiter přihlášení je lokální procesu;
  více workerů nebo replik násobí limity. Pro společný limit je nutná externí
  koordinace; výchozí Docker spouští jeden worker.
- Přenosová fronta prohlížeče má pevný limit 16 000 B (0,5 s PCM). Pokud síť
  nestíhá, mikrofon se zastaví a v historii zůstane upozornění na možnou mezeru.
- Aplikace neukládá audio ani přepisy. Web Speech může audio zpracovávat přes
  službu výrobce prohlížeče; nejde o záruku offline rozpoznávání.

## Web Speech a automatická záloha

Bez podpory `SpeechRecognition` / `webkitSpeechRecognition` se rovnou spustí
Google záloha. Při chybě rozpoznávací služby nebo alespoň 6 s výraznějšího audia
bez výsledků po dobu 8 s se Web Speech zastaví. Již přijaté finaly se nejprve
dopřeloží, potom se automaticky spustí Google. Během přechodu se nic nenahrává
do skryté fronty; rozpracovaná řeč může chybět a v historii zůstane upozornění.
Délka přechodu závisí na frontě finalů a dostupnosti Google, není bezešvý.
Tlačítkem „Zrušit přepnutí“ lze automatický start zálohy zrušit.

Pouhé ticho (`no-speech`) zálohu nespustí; běžně ukončený prázdný běh se
obnoví. Zamítnutí mikrofonu/rozpoznávání se neobchází jinou službou. Chyba
překladače nebo spojení se serverem také není důvod zapínat jiné STT.

Stop u Web Speech zavolá `recognition.stop()` a až 3 s přijímá poslední finaly
do `onend`. Teprve potom ukončí textový vstup a dopřeloží frontu. Pokud zůstane
jen hypotéza, služba ohlásí chybu nebo nedoběhne včas, závěr je označen jako
nepotvrzený. Hypotéza se nikdy nevydává za final. Při přepnutí na zálohu se
statistiky nové relace vynulují; historie zůstane.

Dostupnost i rychlost Web Speech závisí na prohlížeči a jeho službě. Provoz
ověřujte v cílovém prohlížeči; pouhá existence API neznamená funkční backend.
Viz [specifikace Web Speech](https://webaudio.github.io/web-speech-api/).

Rozpoznávač dostává již povolenou mikrofonní `MediaStreamTrack`; nemusí otevírat
druhé snímání. Tuto variantu podporuje
[Chrome od verze 135](https://developer.chrome.com/release-notes/135?hl=en).
Starší implementace nepovinný argument ignorují a použijí výchozí mikrofon.

## Ovládání a živý náhled

Horní lišta obsahuje Start/Stop, fullscreen (ukončení také přes Esc), velikost
textu 18–64 px, světlý/tmavý motiv, sbalení lišty a sledování posledního textu.
Nejde o nastavení překladu: žádná volba modelu, jazyka nebo bufferu se nevrací.
Do localStorage se ukládají pouze tyto prezentační preference, nikdy obsah.

Průběžný EN/RU pár je trvale dostupný nahoře. Pod ním je nejnovější celý final,
starší výsledky následují směrem dolů. Společné záhlaví EN/RU platí pro oba
sloupce na desktopu i mobilu; u jednotlivých vět se hlavičky ani čísla neopakují.
Průběžný panel nejdřív roste dolů do dostupné výšky obrazovky; historii zůstává
malý posouvatelný pruh. Teprve když se text ani tak nevejde, oba jazyky se
automaticky zmenší na největší společnou velikost, která se vejde. Přepočet
reaguje i na změnu okna, fullscreen, sbalení lišty a ruční velikost písma.
Automatické zmenšení platí jen pro interim, nemění uloženou volbu ani finaly.
Po finalu, resetu nebo kratším náhledu se písmo vrací ke zvolené velikosti.
Minimum je 14 px; mimořádně dlouhý text zůstane celý dostupný posouváním,
nikdy se neodřízne. Vypnuté sledování textu se respektuje i při přepočtu.
Rozpracovaný náhled se mění, final přejde do společné historie bez zkracování.
Posunutí historie dolů vypne automatické sledování; nové výsledky potom zachovají
rozečtené místo. Zaškrtnutím „Sledovat text“ se lze vrátit k nejnovějšímu finalu
nahoře. U krátkých promluv může final předběhnout interim.
Pokud interim selže, zůstane viditelné upozornění a mikrofon i finaly pokračují.

Statistiky aktuální relace ukazují délku zachyceného audia, úroveň mikrofonu,
počty zobrazených interim/final párů a počty STT hypotéz/finalů. Časy v ms:

- **STT první:** první PCM rámec → první přepis, včetně úvodního ticha. Web
  Speech se měří v prohlížeči, Google záloha na serveru; nejde o end-to-end
  latenci každé věty.
- **Překlad:** poslední Google překladový požadavek, včetně retry finalu.
- **Čekání:** fronta a slučování před spuštěním vybraného překladu.
- **Připojení:** WebSocket (u zálohy i příprava Google STT), bez povolování
  mikrofonu a startu Web Speech; není RTT.

Rozhraní používá atomic design bez dalšího frameworku: sdílené CSS tokeny,
atomy tlačítek/ikon/statistik, molekuly jazykového páru a lištu jako organismus.
Prezentační JavaScript je oddělený od životního cyklu audia.

## Automatické dělení řeči v Google záloze

Google automatické VAD je vypnuté. Začátky a konce úseků řídí server přes
`activity_start` / `activity_end`, bez přerušování rozpracovaných odpovědí
novým audiem (`NO_INTERRUPTION`). Nejde o ruční tlačítko ani novou volbu v UI.
Server používá levné dělení z energie PCM: po alespoň 2 s
řeči upřednostní 400ms pauzu, po 6 s stačí 200ms pauza. Nejpozději po 12 s
souvislého vstupu odešle `activity_end`, aby nezůstával jeden stále
rostoucí přepis. V hlučném nebo nepřerušovaném vstupu může tento pevný limit
rozdělit i větu; jde o kompromis pro průběžné titulky, nikoli garanci latence
odpovědi Googlu nebo bezchybného rozpoznání.

Žádný audio rámec se nezahazuje, neopakuje ani uměle neprodlužuje. První
`activity_start` předchází již prvnímu vzorku včetně ticha, takže se neodřezává
začátek slova. Před otevřením dalšího úseku server vyčká na `generation_complete`
předchozího úseku, nejvýše 1 s. Zachová pořadí čekajícího audia na témže spojení.
Pokud potvrzení nepřijde, relace se zastaví s `google_timeout`; nepokračuje
potichu s nefunkčním dělením. Celý odesílací krok má navíc timeout 2 s.
Samotný `input_transcription` ani `turn_complete` toto čekání nezkracuje.
Značky jsou výhradně serverové;
prohlížeč dál posílá pouze PCM a Stop. Do historie se jako final ukládá jen
autoritativní `input_transcription`, nikdy spekulativní interim. Toto chování
vychází z [dokumentace ručního VAD](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe#manual-vad-push-to-talk).

Nahrazuje hybridní `audio_stream_end`, při němž se v reálných testech občas
přestaly vracet interimy, zatímco finaly pokračovaly. Pouhé přepnutí značek
bez čekání nestačilo: další `activity_start` mohl předběhnout dokončení úseku
a finaly se zpožďovaly. Dostupnost a latence skutečných výsledků nadále závisí
na Google službě; místní testy nedokážou zaručit její bezchybný provoz.

## Cena a limit relace

Při Web Speech se platí jen textové překlady v Gemini, nikoli Google Live STT.
Následující cena audia se uplatní pouze u zálohy.
Live Transcribe stojí orientačně **$0.009 za minutu audia**, tedy přibližně
**$0.09 za plnou desetiminutovou relaci**. K tomu se přičte textový překlad:

| Model | Vstup / 1M tokenů | Výstup / 1M tokenů |
|---|---:|---:|
| `gemini-3.1-flash-lite` (interim) | $0.25 | $1.50 |
| `gemini-3.5-flash-lite` (final) | $0.30 | $2.50 |

Textová část bývá proti ceně audia malá, ale její přesná cena závisí na délce
řeči a počtu změn interim přepisu. Aktuální ceny vždy ověřte v
[oficiálním ceníku Gemini API](https://ai.google.dev/gemini-api/docs/pricing).

Google omezuje jednu Live Transcribe session na **10 minut**. Server proto po
9 minutách a 45 sekundách zastaví vstup a zahájí dokončování. Stejný ochranný
limit platí pro textové relace Web Speech. Novou relaci lze
spustit po uzavření té předchozí.

Následující pravidla Stop platí pro **Google audio zálohu** (Web Speech viz výše).
Po stisku Stop se ihned vypne mikrofon, tlačítko přejde na „Dokončuji…“ a
už přijaté finaly se dopřeloží. Zcela prázdná relace (žádný odeslaný vzorek ani
přijatý přepis) skončí rovnou bez `activity_end` a bez varování. Rozhoduje
skutečný vstup, nikoli zaokrouhlený čas `00:00` nebo detekce ticha.

U neprázdné relace Stop není potvrzení poslední promluvy. Odeslání posledního `activity_end` má timeout
2 s a server potom přijímá výsledky napříč všemi tahy až 5 s, nikoli pouze do
prvního finalu. `generation_complete` slouží jen k oddělení průběžných úseků,
nikdy nepotvrzuje kompletní zpracování celého streamu při Stop.
[Live Transcribe](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
nedokumentuje potvrzení zpracování všech odeslaných vzorků. Pokud Google po Stop
spojení sám řádně neukončí, aplikace po uplynutí této lhůty ohlásí
`transcription_incomplete`, dokončí již přijaté finální překlady a uzavře spojení
kódem 1011. Neodešle úspěšné `ended`. Toto upozornění tedy může přijít i tehdy,
když se poslední překlad zobrazil; úplnost nelze potvrdit pouhým tichem nebo
jedním finalem.

Nepotvrzený závěr se v UI zobrazuje jako kompaktní žluté upozornění, nejvýše
jednou za relaci, se stavem „Zastaveno · závěr nepotvrzen“. Hotové překlady
zůstávají viditelné. Po ukončení se skryje prázdné tělo interim panelu;
při dalším spuštění se opět otevře. Samotné zavření spojení bez `ended` po
odeslaném audiu nepovažuje prohlížeč za úspěšné dokončení Stop.

Pokud po doběhnutí přijatých finalů zbývá průběžný náhled, uloží se do žlutého
upozornění jako **nepotvrzený překlad**, aby nezmizel. Nezvyšuje počet finalů
a není vydáván za dokončený výsledek. Pozdější final přijatý ještě během
dokončování náhled nahradí, takže se nearchivuje dvakrát.

Odpojení prohlížeče ruší rozpracované placené požadavky i během dokončování,
bez dalšího retry. Ukončování má navíc celkový deadline zahrnující odeslání Stop,
příjem a maximální dobu dokončení omezené fronty finalů. Chybějící finální
překlad zůstává označený v historii i po dalších úspěšných výsledcích; při nové
relaci se čistí pouze spekulativní interim, nikoli historie.

## Spuštění

Produkční a CI runtime: Python 3.12. Dále je potřeba Google Gemini API klíč;
pro kompletní vývojové testy také Node.js 24 (bez npm závislostí).

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
{"type":"telemetry","kind":"interim","stage":"translating","hypotheses":3,"finals_received":0,"first_stt_ms":850,"queue_ms":350}
{"type":"interim","en":"Hello","ru":"Здравствуйте"}
{"type":"telemetry","kind":"interim","stage":"translated","hypotheses":3,"finals_received":0,"first_stt_ms":850,"queue_ms":350,"translation_ms":450}
{"type":"final","en":"Hello.","ru":"Здравствуйте."}
{"type":"ended"}
```

Chyby mají stabilní kód bez detailů nebo API klíče:

```json
{"type":"error","code":"google_unavailable","recoverable":true}
```

Po terminální chybě mohou dříve přijaté finaly ještě doběhnout před uzavřením.
`translation_failed` označuje jeden chybějící final a relaci neukončuje.
`interim_translation_failed` označuje neúspěšný náhled, neukončuje mikrofon a
nemá retry. `transcription_incomplete` označuje neověřený konec vstupu.
`telemetry` neobsahuje přepis ani klíče; časy, které nejsou změřené, chybí nebo
jsou `null`. Zprávy EN/RU zůstávají společné a beze změny.

## Diagnostické logy v Coolify

Kontejner zapisuje při výchozím spuštění řádky `translator {JSON}` přes logger
Uvicornu (INFO, chyby WARNING). Nevyžadují nové env vars ani SDK DEBUG logování.
Každá relace má náhodné serverové `session_id`, každý vybraný překlad `work_id`.
Čas `timestamp` je vždy UTC; čas zobrazený Coolify může používat jinou zónu.

- `session_open`, `session_ready`, `session_stop_requested`,
  `session_disconnect`, `session_close_requested`, `session_close`: životní
  cyklus, cesta `web_speech` / `google_audio`, close kód a výsledek ukončení.
- `stt_progress`: počty přijatých přepisů a délka vstupu ve znacích. První přepis
  a každý final jsou viditelné; interim souhrn nejvýše jednou za 5 s.
- `translation_started`, `translation_succeeded`, `translation_failed`,
  `translation_cancelled`: vybraný model, čekání ve frontě, délka požadavku,
  čekající finaly a zrušení například kvůli novému finalu nebo ukončení relace.
- `translation_attempt_failed`, `translation_retry`: jednotlivé pokusy,
  HTTP status (např. 429/503), timeout nebo validace JSON. U neplatné odpovědi
  se může objevit bezpečný `finish_reason`, např. `MAX_TOKENS`.
- `session_error`: stejný bezpečný chybový kód jako v prohlížeči. Závěrečný
  `session_close` obsahuje počty přijatých, spuštěných, úspěšných, neúspěšných
  a zrušených překladů i retry. `completed` znamená dokončené ukončení relace,
  nikoli záruku úspěchu všech překladů; zkontrolujte také počty selhání.

`first_transcript_ms_since_session` měří čas od otevření serverové relace,
nikoli od začátku řeči. `first_stt_ms` je na serveru dostupné pouze u Google
audio zálohy. `audio_bytes` počítá PCM pokusně předané Google záloze (nepotvrzuje
jeho zpracování); na Web Speech cestě je vždy nula. Úspěšný překlad v logu
znamená odeslání do WebSocketu, nikoli potvrzené vykreslení na klientovi.
`timeout_ms` je limit jednoho pokusu; `translation_ms` u finalu zahrnuje
i případný retry. Zrušený požadavek negarantuje nulovou cenu na straně Googlu.

Logy neobsahují přepisy, překlady, audio, cookies, klíče, celé providerové
odpovědi ani texty výjimek. Povolené jsou jen pevné kódy a číselné metriky.
Chyby browserové služby Web Speech a kliknutí na Stop se neposílají novým
diagnostickým protokolem: server vidí přepisy, doručený Stop a odpojení,
neumí ale z nich s jistotou určit důvod chyby uvnitř prohlížeče. Pro tento
případ je stále důležitý screenshot hlášky/statistik nebo konzole prohlížeče.
Historické události před nasazením této diagnostiky nelze zpětně doplnit.

## Docker

```bash
docker build -t czech-live-translator .
docker run --rm --env-file .env -p 8000:8000 czech-live-translator
```

Kontejner běží pod neprivilegovaným uživatelem. `GET /health/live` ověřuje běh
procesu a `GET /health/ready` vrací 200 pouze s použitelnou Google a auth
konfigurací. Docker `HEALTHCHECK` používá `/health/ready`.

V Coolify vyberte build pack **Dockerfile**, soubor `/Dockerfile` a port `8000`.
Pokud nastavujete vlastní Healthcheck v Coolify, použijte typ **CMD** a příkaz
`python -m app.healthcheck` (interval 30 s, timeout 5 s, start period 10 s,
3 pokusy). Kontrola používá pouze Python a runtime `PORT`; HTTP kontrola z UI
Coolify vyžaduje `curl` nebo `wget`, které tento malý image neobsahuje.

Za reverzní proxy povolte WebSocket upgrade, používejte HTTPS a správně nastavte
`FORWARDED_ALLOW_IPS` i `ALLOWED_ORIGINS`.

## Kontroly

```bash
python -m compileall app tests
pytest
pytest --cov=app --cov-report=term-missing
node --test tests/frontend-runtime.test.cjs
pip-audit -r requirements.txt
docker build --check .
docker build .
```

CI provádí audit runtime závislostí, pytest (včetně Node regresí nad skutečným
prezentačním a audio skriptem s náhradami browser API), validaci Dockerfile a sestavení
produkčního obrazu. Tyto testy nevolají placené Google API a nenahrazují ověření
skutečného mikrofonu a sítě v prohlížeči.
