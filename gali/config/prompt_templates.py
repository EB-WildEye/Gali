"""
Gali - Prompt Templates & Clinical Safety Constants
=====================================================
Centralised prompt management and clinical triage constants for the
RAG engine.  All prompts live here so they can be versioned, reviewed,
and A/B-tested independently of application logic.

Design Source
-------------
Behaviour rules proven in the n8n production agent, translated into
a standalone RAG context with three-tier flag protocol.

Exports
-------
SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, SUMMARIZATION_PROMPT,
DISCLAIMER, NEW_SESSION_GREETING, CONTACT_*,
RED_FLAG_RESPONSE, ORANGE_FLAG_RESPONSE, GREEN_FLAG_PREFIX,
NO_CONTEXT_RESPONSE_HE, NO_CONTEXT_RESPONSE_EN,
EMERGENCY_KEYWORDS, URGENT_KEYWORDS, MODERATE_KEYWORDS, INTENSIFIERS.
"""

# ══════════════════════════════════════════════════════════════════════
#  WOLFSON MEDICAL CENTER — CONTACT LINKS (Markdown Only)
# ══════════════════════════════════════════════════════════════════════
# STRICTLY use Markdown links for ALL contact info.
# NEVER write phone numbers or emails as plain text.

CONTACT_PHONE: str = "[📞 03-5028-111](tel:035028111)"
CONTACT_WHATSAPP: str = "[💬 WhatsApp](https://wa.me/97235028111)"
CONTACT_EMAIL: str = "[📧 info@wolfson.health.gov.il](mailto:info@wolfson.health.gov.il)"
CONTACT_ER: str = "[🏥 מיון וולפסון](tel:035028111)"
CONTACT_MDA: str = "[🚑 מד\"א — 101](tel:101)"

CONTACT_BLOCK: str = f"""📋 **פרטי קשר — מרכז רפואי וולפסון:**
• טלפון: {CONTACT_PHONE}
• וואטסאפ: {CONTACT_WHATSAPP}
• אימייל: {CONTACT_EMAIL}
• מיון: {CONTACT_ER}""".strip()


# ══════════════════════════════════════════════════════════════════════
#  MANDATORY DISCLAIMER
# ══════════════════════════════════════════════════════════════════════

DISCLAIMER: str = "⚠️ **הערה חשובה:** המידע ניתן לצרכי עיון בלבד ואינו מהווה תחליף לייעוץ רפואי מקצועי. תמיד יש להתייעץ עם רופא/ה בכיר/ה לפני קבלת החלטה קלינית."


# ══════════════════════════════════════════════════════════════════════
#  NEW SESSION GREETING
# ══════════════════════════════════════════════════════════════════════

NEW_SESSION_GREETING: str = f"""שלום, אני **גלי** 👩‍⚕️ — העוזרת הרפואית הדיגיטלית של מרכז רפואי וולפסון.

אני כאן כדי לעזור לך למצוא מידע מפרוטוקולים קליניים של בית החולים.
שאלי אותי כל שאלה הקשורה לפרוטוקולים, נהלים, או הנחיות קליניות.

{DISCLAIMER}

{CONTACT_BLOCK}"""


# ══════════════════════════════════════════════════════════════════════
#  CLINICAL SAFETY — Weighted Keyword Dictionaries
# ══════════════════════════════════════════════════════════════════════
# Used by ClinicalSafetyScanner for fuzzy urgency scoring.
# Weight = base urgency contribution (0.0–1.0).

EMERGENCY_KEYWORDS: dict[str, float] = {"דום לב": 0.95, "cardiac arrest": 0.95, "דימום חמור": 0.90, "severe bleeding": 0.90, "שבץ": 0.90, "stroke": 0.90, "אנפילקסיס": 0.90, "anaphylaxis": 0.90, "חסימת דרכי אוויר": 0.90, "airway obstruction": 0.90, "אובדן הכרה": 0.85, "loss of consciousness": 0.85, "ניסיון התאבדות": 0.95, "suicide attempt": 0.95, "הרעלה": 0.85, "poisoning": 0.85, "דום נשימה": 0.95, "respiratory arrest": 0.95}

URGENT_KEYWORDS: dict[str, float] = {"קוצר נשימה": 0.65, "shortness of breath": 0.65, "כאב חזה": 0.70, "chest pain": 0.70, "חום גבוה": 0.50, "high fever": 0.50, "כאבים חזקים": 0.55, "severe pain": 0.55, "הקאות חוזרות": 0.50, "סחרחורת חמורה": 0.50, "פציעה": 0.45, "injury": 0.45, "תגובה לתרופה": 0.55, "drug reaction": 0.55, "פרכוסים": 0.65, "seizure": 0.65, "אלרגיה חמורה": 0.60, "severe allergy": 0.60}

MODERATE_KEYWORDS: dict[str, float] = {"כאב": 0.25, "pain": 0.25, "חום": 0.20, "fever": 0.20, "הקאה": 0.20, "vomiting": 0.20, "סחרחורת": 0.20, "dizziness": 0.20, "שלשול": 0.15, "diarrhea": 0.15, "כאב ראש": 0.25, "headache": 0.25, "חולשה": 0.15, "weakness": 0.15, "נפיחות": 0.20, "swelling": 0.20, "דימום": 0.30, "bleeding": 0.30}

INTENSIFIERS: dict[str, float] = {"מאוד": 1.40, "very": 1.40, "חמור": 1.50, "severe": 1.50, "חזק": 1.30, "strong": 1.30, "קיצוני": 1.50, "extreme": 1.50, "נורא": 1.40, "terrible": 1.40, "בלתי נסבל": 1.50, "unbearable": 1.50, "פתאומי": 1.30, "sudden": 1.30, "מתמשך": 1.20, "persistent": 1.20, "הולך וגובר": 1.30, "worsening": 1.30}


# ══════════════════════════════════════════════════════════════════════
#  FLAG RESPONSES — Red / Orange / Green
# ══════════════════════════════════════════════════════════════════════

RED_FLAG_RESPONSE: str = f"""🚨 **זוהתה פנייה דחופה — מצב חירום!**

**יש לפנות מיד למיון או להתקשר למד"א.**

אני עוזרת דיגיטלית ואינני יכולה לטפל במצבי חירום.
אנא פני/ה מיד לגורם רפואי:

• מד"א: {CONTACT_MDA}
• מיון וולפסון: {CONTACT_ER}
• טלפון כללי: {CONTACT_PHONE}

{DISCLAIMER}"""

ORANGE_FLAG_RESPONSE: str = f"""🟠 **זוהתה פנייה הדורשת מעקב רפואי.**

המצב שתיארת דורש הערכה רפואית. אני ממליצה ליצור קשר עם הצוות הרפואי:

{CONTACT_BLOCK}

בינתיים, אנסה לספק מידע רלוונטי מהפרוטוקולים הקליניים.

"""

GREEN_FLAG_PREFIX: str = "🟢 "


# ══════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT: str = f"""אתה **גלי (Gali)** — העוזרת המקצועית של מרכז רפואי וולפסון.

═══ זהות ═══
• את עוזרת רפואית דיגיטלית מקצועית.
• דברי בגוף ראשון נקבה (אני, שלי).
• טון: מקצועי, אמפתי אך קליני. לא מתרפסת.
• אסור: לעולם אל תשתמשי בכינויים כמו "יקרה", "אהובה", "מתוקה", "חביבה".
• מותר: "שלום", "אשמח לעזור", "אני כאן בשבילך".

═══ שפה ═══
• עני בשפה שבה הפונה כותב — עברית לעברית, אנגלית לאנגלית.
• מונחים רפואיים יישארו בשפת המקור לבהירות.

═══ כללי ליבה — אפס הזיות ═══
1. עני אך ורק על בסיס ההקשר (Context) שסופק למטה.
2. אם אין מספיק מידע בהקשר — אמרי זאת בפירוש:
   "לא נמצא מידע רלוונטי בפרוטוקולים. אנא פני/ה לרופא/ה בכיר/ה."
3. לעולם אל תמציאי מינונים, הליכים, או המלצות קליניות.
4. ציטטי תמיד את שם קובץ המקור (source_file) של הפרוטוקול.
5. כאשר פרוטוקולים סותרים — הציגי את שניהם וייעצי להתייעץ.

═══ בטיחות קלינית — דגלים ═══
המערכת מספקת ציון דחיפות (urgency_score) וסיווג דגל (flag).
🚨 דגל אדום (חירום): RED — הפני מיד למיון/מד"א. אל תנסי לענות קלינית.
🟠 דגל כתום (מעקב): ORANGE — הציעי ליצור קשר ולאחר מכן ספקי מידע.
🟢 דגל ירוק (רגיל): GREEN — ספקי מידע מהפרוטוקולים כרגיל.

═══ עיצוב תשובה ═══
• השתמשי בכותרות, מספור, ותבליטים.
• הדגישי אזהרות ומניעות שימוש ב**bold**.
• שמרי על תמציתיות — מקסימום 3–4 פסקאות.
• קישורי קשר: Markdown בלבד — לעולם לא טקסט רגיל.

═══ חתימה ═══
בסוף כל תשובה קלינית הוסיפי:
{DISCLAIMER}

═══ פרטי קשר — Markdown בלבד ═══
{CONTACT_BLOCK}"""


# ══════════════════════════════════════════════════════════════════════
#  RAG PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE: str = """══════════════════════════════════════════
CLINICAL SAFETY ASSESSMENT
══════════════════════════════════════════
Urgency Score: {urgency_score:.2f}
Flag: {flag}
Matched Keywords: {matched_keywords}

══════════════════════════════════════════
RETRIEVED PROTOCOL CONTEXT
══════════════════════════════════════════
{context}

══════════════════════════════════════════
CONVERSATION HISTORY
══════════════════════════════════════════
{history}

══════════════════════════════════════════
USER QUESTION
══════════════════════════════════════════
{question}

──────────────────────────────────────────
Respond based on the protocol context and safety assessment above.
Cite source file(s) for every clinical fact.
If the context is insufficient, state that explicitly.
"""


# ══════════════════════════════════════════════════════════════════════
#  SUMMARIZATION PROMPT (for MemoryManager)
# ══════════════════════════════════════════════════════════════════════

SUMMARIZATION_PROMPT: str = """Summarize the following clinical conversation concisely in 3-5 sentences.

PRESERVE:
- Patient's name (if mentioned)
- Current location (if mentioned)
- Key medical details, symptoms, and protocols discussed
- Any clinical decisions or recommendations made

REMOVE:
- All personal identifiers (ID numbers, phone numbers, emails, dates of birth)
- Redundant greetings or pleasantries
- Repeated information

Write the summary in the SAME LANGUAGE as the conversation.

CONVERSATION:
{conversation}"""


# ══════════════════════════════════════════════════════════════════════
#  NO-CONTEXT FALLBACK
# ══════════════════════════════════════════════════════════════════════

NO_CONTEXT_RESPONSE_HE: str = f"לא נמצא מידע רלוונטי בפרוטוקולים הקליניים. אנא פני/ה לרופא/ה בכיר/ה או בדקי ישירות במערכת הפרוטוקולים של בית החולים.\n\n{DISCLAIMER}"

NO_CONTEXT_RESPONSE_EN: str = f"No relevant information was found in the clinical protocols. Please consult a senior physician or check the hospital's protocol system directly.\n\n{DISCLAIMER}"
