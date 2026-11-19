"""Ukrainian display layer for the soldier report form only.

How it works: every visible English string on the form goes through t(). When the soldier
picks УКР, t() swaps in the Ukrainian text from UK below. Only the SCREEN changes: the
answers kept in session state and written to the database stay the English values, so the
analyst side and the shared v3 database never see Ukrainian.

The language choice lives in this browser session only (st.session_state["lang"]). Nothing
about it is stored, and every other page (start page, database, insights, graphs) stays English.

Translations are a first pass for the prototype. Military terms should be checked by a
native-speaking Ukrainian partner before anyone relies on them.
"""
import streamlit as st

LANGS = {"en": "EN", "uk": "УКР"}

UK = {
    # ----- page, steps, navigation ------------------------------------------------------
    "Daily engagement report": "Щоденний звіт про бойове зіткнення",
    "Level 1 · Tactical. About 2 minutes. Only answers marked * are required.":
        "Рівень 1 · Тактичний. Близько 2 хвилин. Обов'язкові лише відповіді, позначені *.",
    "Unit & task": "Підрозділ і завдання",
    "Conditions": "Умови",
    "Platforms & ratings": "Засоби та оцінки",
    "Malfunctions & result": "Несправності та результат",
    "Review & submit": "Перевірка та надсилання",
    "Back to start": "Назад на початок",
    "Quick mode: required answers only": "Швидкий режим: лише обов'язкові відповіді",
    "Shows only the required questions on one page.":
        "Показує лише обов'язкові питання на одній сторінці.",
    "{done} of {total} required answers done": "Виконано {done} з {total} обов'язкових відповідей",
    "Back: {s}": "Назад: {s}",
    "Next: {s}": "Далі: {s}",
    "Submit report": "Надіслати звіт",
    "Your answers are kept while you move between steps.":
        "Ваші відповіді зберігаються, поки ви переходите між кроками.",
    "Can't submit yet. Use the red Fix buttons above to finish the missing answers.":
        "Поки що не можна надіслати. Натисніть червоні кнопки «Виправити» вище, щоб заповнити пропущене.",
    "Some required answers are missing. They're marked in red above.":
        "Бракує деяких обов'язкових відповідей. Їх позначено червоним вище.",
    "Required.": "Обов'язково.",
    "Choose…": "Оберіть…",
    "Language": "Мова",

    # ----- report sent ----------------------------------------------------------------------
    "Report #{rid} sent. Thank you.": "Звіт №{rid} надіслано. Дякуємо.",
    "Return to start": "Повернутися на початок",
    "File another report": "Подати ще один звіт",
    "Your unit": "Ваш підрозділ",
    "Secure upload": "Захищене передавання",
    "Central database": "Центральна база даних",
    "Analyst cell": "Аналітична група",
    "Field units only file reports. They never see the database or the analysis.":
        "Польові підрозділи лише подають звіти. Вони не бачать ні бази даних, ні аналізу.",

    # ----- step 1: unit & task --------------------------------------------------------------
    "Report date": "Дата звіту",
    "Unit type": "Тип підрозділу",
    "Primary task": "Основне завдання",
    "Unit ID (optional)": "Позначення підрозділу (необов'язково)",
    "e.g. U-417": "напр. U-417",
    "Required: unit type.": "Обов'язково: тип підрозділу.",
    "Required: primary task.": "Обов'язково: основне завдання.",

    # ----- step 2: conditions & adversary ---------------------------------------------------
    "Location & conditions (optional)": "Місце та умови (необов'язково)",
    "Fill in what you know. Click a selected answer again to clear it.":
        "Заповніть те, що знаєте. Натисніть вибрану відповідь ще раз, щоб її скасувати.",
    "Sector / location": "Сектор / місце",
    "Enemy EW / jamming": "РЕБ противника / глушіння",
    "“None” means no jamming; leave blank if you don't know.":
        "«Немає» означає, що глушіння не було; залиште порожнім, якщо не знаєте.",
    "Weather": "Погода",
    "Time of day": "Час доби",
    "Terrain": "Місцевість",
    "Adversary (optional)": "Противник (необов'язково)",
    "Add adversary details": "Додати дані про противника",
    "Off = nothing recorded for the adversary (not even “Unknown”).":
        "Вимкнено = про противника нічого не записується (навіть «Невідомо»).",
    "Enemy type": "Тип противника",
    "Estimated enemy strength": "Орієнтовна чисельність противника",
    "Enemy air defense": "ППО противника",
    "Enemy adaptation": "Адаптація противника",
    "Skipped. Turn on to describe the enemy force.":
        "Пропущено. Увімкніть, щоб описати сили противника.",

    # ----- step 3: platforms ----------------------------------------------------------------
    "Which platforms did you use?": "Які засоби ви застосовували?",
    "Platforms employed": "Застосовані засоби",
    "Type a name or a category, e.g. Javelin, Puma, FPV, Anti-Armor":
        "Введіть назву або категорію, напр. Javelin, Puma, FPV, Протитанкові",
    "Pick every platform you used. Not listed? Type its name and press Enter "
    "to add it as a custom platform.":
        "Оберіть усі засоби, які ви застосовували. Немає в списку? Введіть назву й натисніть "
        "Enter, щоб додати власний засіб.",
    "Required: pick at least one platform.": "Обов'язково: оберіть щонайменше один засіб.",
    "{n} selected. Each one gets its own rating card below.":
        "Обрано: {n}. Кожен засіб має власну картку оцінки нижче.",
    "Nothing selected yet.": "Ще нічого не обрано.",
    "BLP stylized model · illustrative": "Стилізована модель BLP · ілюстрація",
    "How did it perform?": "Як він спрацював?",
    "Effectiveness (overall)": "Ефективність (загалом)",
    "Required: rate effectiveness from 1 to 5.": "Обов'язково: оцініть ефективність від 1 до 5.",
    "TRAITS (OPTIONAL) · 1 = POOR, 5 = BEST":
        "ХАРАКТЕРИСТИКИ (НЕОБОВ'ЯЗКОВО) · 1 = ПОГАНО, 5 = НАЙКРАЩЕ",
    "Ease of maintenance": "Простота обслуговування",
    "5 = easy": "5 = легко",
    "Spare parts availability": "Наявність запчастин",
    "5 = readily available": "5 = легко дістати",
    "Platform survivability": "Живучість засобу",
    "5 = survives well": "5 = добре вцілює",
    "Crew survivability": "Живучість екіпажу",
    "5 = crew survives well": "5 = екіпаж добре вцілює",
    "Lethality": "Вражаюча дія",
    "5 = very lethal": "5 = дуже висока",
    "How hard is it to hit": "Наскільки важко в нього влучити",
    "5 = very hard to hit": "5 = дуже важко влучити",
    "Field-modified by the unit?": "Доопрацьовано підрозділом у польових умовах?",

    # ----- step 4: malfunctions & result ----------------------------------------------------
    "Malfunctions (whole engagement)": "Несправності (за весь бій)",
    "Jams / malfunctions observed": "Помічені затримки / несправності",
    "Pick “None observed” or every problem you saw.":
        "Оберіть «Не помічено» або всі проблеми, які ви бачили.",
    "Required: pick “None observed” or a malfunction.":
        "Обов'язково: оберіть «Не помічено» або несправність.",
    "▸ SHOWN BECAUSE YOU REPORTED A MALFUNCTION": "▸ ПОКАЗАНО, БО ВИ ПОВІДОМИЛИ ПРО НЕСПРАВНІСТЬ",
    "Impact on the mission": "Вплив на виконання завдання",
    "Mission result": "Результат завдання",
    "Mission outcome": "Підсумок завдання",
    "Required: pick the mission outcome.": "Обов'язково: оберіть підсумок завдання.",
    "What most limited the result? (optional)": "Що найбільше обмежило результат? (необов'язково)",
    "Notes (optional)": "Примітки (необов'язково)",
    "Anything that didn't fit above.": "Усе, що не вмістилося вище.",
    "Objective fully achieved.": "Мету повністю досягнуто.",
    "Part of the objective achieved.": "Мету досягнуто частково.",
    "Objective not achieved.": "Мети не досягнуто.",
    "Mission stopped early or not attempted.": "Завдання припинено достроково або не розпочато.",

    # ----- step 5: review -------------------------------------------------------------------
    "Review before you submit": "Перевірте перед надсиланням",
    "{n} required answer(s) still missing.": "Бракує обов'язкових відповідей: {n}.",
    "Fix: {s}": "Виправити: {s}",
    "At least one platform": "Щонайменше один засіб",
    "Effectiveness for {p}": "Ефективність: {p}",
    "Malfunction check": "Перевірка несправностей",
    "Missing": "Бракує",
    "UNIT & TASK": "ПІДРОЗДІЛ І ЗАВДАННЯ",
    "LOCATION & CONDITIONS": "МІСЦЕ ТА УМОВИ",
    "ADVERSARY": "ПРОТИВНИК",
    "MALFUNCTIONS & RESULT": "НЕСПРАВНОСТІ ТА РЕЗУЛЬТАТ",
    "NOTES": "ПРИМІТКИ",
    "PLATFORMS": "ЗАСОБИ",
    "Date": "Дата",
    "Unit": "Підрозділ",
    "Unit ID": "Позначення",
    "Task": "Завдання",
    "Sector": "Сектор",
    "Enemy EW": "РЕБ противника",
    "Strength": "Чисельність",
    "Air defense": "ППО",
    "Adaptation": "Адаптація",
    "Skipped (nothing recorded).": "Пропущено (нічого не записано).",
    "Outcome": "Підсумок",
    "Limiting factor": "Обмежувальний чинник",
    "Malfunctions": "Несправності",
    "Impact": "Вплив",
    "Effectiveness missing": "Немає оцінки ефективності",
    "Field-modified": "Доопрацьовано",
    "Maintenance": "Обслуговування",
    "Spare parts": "Запчастини",
    "Platform survival": "Живучість засобу",
    "Crew survival": "Живучість екіпажу",
    "Hard to hit": "Важко влучити",

    # ----- answer choices (display only; stored values stay English) ------------------------
    "Other": "Інше",
    "Unknown": "Невідомо",
    "None": "Немає",
    "Low": "Низький",
    "Medium": "Середній",
    "High": "Високий",
    "No": "Ні",
    "Yes": "Так",
    # unit types
    "Drone crew": "Екіпаж БпЛА",
    "Outpost": "Опорний пункт",
    "Infantry squad": "Піхотне відділення",
    "Artillery section": "Артилерійський розрахунок",
    "EW team": "Група РЕБ",
    "Mechanized platoon": "Механізований взвод",
    # tasks
    "Reconnaissance": "Розвідка",
    "Strike": "Ураження",
    "Defense": "Оборона",
    "Counter-drone": "Протидія БпЛА",
    "Resupply / logistics": "Постачання / логістика",
    "EW / jamming": "РЕБ / глушіння",
    # sectors
    "Kharkiv": "Харків",
    "Kupiansk": "Куп'янськ",
    "Lyman / Siversk": "Лиман / Сіверськ",
    "Kramatorsk / Sloviansk": "Краматорськ / Слов'янськ",
    "Chasiv Yar / Toretsk": "Часів Яр / Торецьк",
    "Pokrovsk": "Покровськ",
    "Kurakhove / Vuhledar": "Курахове / Вугледар",
    "Huliaipole": "Гуляйполе",
    "Orikhiv / Zaporizhzhia": "Оріхів / Запоріжжя",
    "Kherson / Dnipro River": "Херсон / річка Дніпро",
    "Black Sea / Odesa": "Чорне море / Одеса",
    "Northern border (Sumy)": "Північний кордон (Суми)",
    "Rear area / deep strike": "Тил / глибоке ураження",
    # weather, time, terrain
    "Clear": "Ясно",
    "Overcast": "Хмарно",
    "Rain": "Дощ",
    "Fog": "Туман",
    "Snow": "Сніг",
    "High wind": "Сильний вітер",
    "Day": "День",
    "Night": "Ніч",
    "Dawn / dusk": "Світанок / сутінки",
    "Open field": "Відкрите поле",
    "Forest / treeline": "Ліс / лісосмуга",
    "Urban": "Міська забудова",
    "Trench line": "Лінія окопів",
    "River / wetland": "Річка / заболочена місцевість",
    # adversary
    "Infantry": "Піхота",
    "Mechanized / armor": "Механізовані / бронетехніка",
    "Artillery": "Артилерія",
    "UAS operators": "Оператори БпЛА",
    "EW unit": "Підрозділ РЕБ",
    "Mixed": "Змішані сили",
    "None observed": "Не помічено",
    "Tactic change": "Зміна тактики",
    "Countermeasure to platform": "Протидія засобу",
    # platform status & effectiveness
    "Performed as expected": "Спрацював, як очікувалося",
    "Degraded but usable": "Погіршений, але придатний",
    "Failed to perform": "Не спрацював",
    "Lost / destroyed": "Втрачено / знищено",
    "1 · Very poor": "1 · Дуже погано",
    "2 · Poor": "2 · Погано",
    "3 · Mixed": "3 · Посередньо",
    "4 · Good": "4 · Добре",
    "5 · Excellent": "5 · Відмінно",
    # malfunctions & impact
    "Jam / stoppage": "Затримка / заклинювання",
    "Misfire / dud": "Осічка / нерозрив",
    "Comms / data-link loss": "Втрата зв'язку / каналу даних",
    "EW / GPS interference": "Перешкоди РЕБ / GPS",
    "Guidance / targeting failure": "Збій наведення / цілевказання",
    "Power / battery": "Живлення / акумулятор",
    "Sensor / optics": "Сенсори / оптика",
    "Mechanical failure": "Механічна несправність",
    "Software / firmware": "Програмне забезпечення / прошивка",
    "No impact": "Без впливу",
    "Minor": "Незначний",
    "Degraded mission": "Виконання ускладнено",
    "Mission abort": "Завдання зірвано",
    "Platform lost": "Засіб втрачено",
    # outcomes & drivers
    "Success": "Успіх",
    "Partial": "Частково",
    "Partial success": "Частковий успіх",
    "Aborted": "Скасовано",
    "Failure": "Невдача",
    "Equipment performance / malfunction": "Робота техніки / несправність",
    "Enemy EW / countermeasures": "РЕБ / протидія противника",
    "Enemy fires / attrition": "Вогонь противника / втрати",
    "Supply / sustainment shortage": "Брак постачання / забезпечення",
    "Weather / terrain": "Погода / місцевість",
    "Training / procedures": "Підготовка / процедури",
    "Intelligence / targeting error": "Помилка розвідки / цілевказання",
    # platform categories & origins (platform names themselves stay as-is)
    "Air Defense": "ППО",
    "Anti-Armor": "Протитанкові засоби",
    "Armored / Vehicles": "Бронетехніка / автомобілі",
    "Artillery / Mortars": "Артилерія / міномети",
    "Counter-UAS": "Протидія БпЛА",
    "EW / C2 / Comms": "РЕБ / управління / зв'язок",
    "FPV / Small Strike UAS": "FPV / малі ударні БпЛА",
    "Loitering / Strike UAS": "Баражуючі / ударні БпЛА",
    "Naval / USV": "Морські засоби / БЕК",
    "Precision Munitions": "Високоточні боєприпаси",
    "Recon / ISR UAS": "Розвідувальні БпЛА",
    "Rockets / Missiles": "Ракети / реактивні снаряди",
    "Small Arms / Crew-Served": "Стрілецька та групова зброя",
    "Unlisted / custom": "Не в списку / власний",
    "U.S.": "США",
    "Ukraine": "Україна",
    "Other partner": "Інший партнер",
    "Custom": "Власний",
}


def active():
    """True only while the report form is drawing and the soldier picked УКР."""
    return bool(st.session_state.get("_tr_on"))


def t(text, **fmt):
    """Translate one display string (or template) when Ukrainian is on; else return it as-is."""
    if text is None:
        return text
    s = str(text)
    if active():
        s = UK.get(s, s)
    return s.format(**fmt) if fmt else s


def set_lang():
    st.session_state["lang"] = st.session_state.get("lang_pick") or "en"


def language_switch():
    """EN / УКР toggle pinned to the top-right corner of the report form."""
    cur = st.session_state.get("lang", "en")
    st.session_state["lang_pick"] = cur
    with st.container(key="lang_corner"):
        st.segmented_control("Language", list(LANGS), format_func=LANGS.get, key="lang_pick",
                             on_change=set_lang, label_visibility="collapsed",
                             selection_mode="single", required=True)
    st.session_state["_tr_on"] = cur == "uk"
