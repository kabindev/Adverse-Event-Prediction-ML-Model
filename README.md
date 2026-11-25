# Adverse-Event-Prediction-ML-Model
# Understanding My Project — In Plain Language 
This isn't a report dump. It's the *reasoning* — for every step, ask "why did I need to do this?" and 
the answer is right there. Read it like a story: one problem leads to the next decision. --- 
## The Big Picture First 
Imagine a hospital or drug company gets thousands of reports like *"Patient took Drug X, felt dizzy."* 
Someone has to read every single one and decide: Is this serious? Is the drug actually causing it? Are 
we suddenly getting way more of these reports than usual? 
Doing that by hand doesn't scale. So the project asks one simple question: 
> **"Can we build a system that automatically ingests these reports, cleans them up, and uses 
machine learning to help flag the risky ones — while being honest about how well it actually 
works?"** 
Everything else in the project is just the steps needed to answer that question properly. --- 
## The Journey of One Data Point (this is the whole architecture in one paragraph) 
A report starts as a raw file → lands in **AWS S3** (cheap storage, like a mailbox) → gets pulled into 
**Snowflake** (the database/warehouse) into a "RAW" area untouched → gets cleaned up visually 
in **Dataiku** (fixing typos, dates, duplicates) → gets organized into a proper **star schema** (a 
clean, structured set of tables) → and *then* — and only then — machine learning models are 
trained on it inside Snowflake, and the results get written back as new tables. 
**Why this order matters**: you cannot train a good model on messy data. Garbage in, garbage out. 
So 3 of the 5 stages are just about getting the data trustworthy *before* any ML happens. That's a 
deliberate, defensible design choice — mention it if asked "why so many stages?" 
--- 
## Stage 1: Getting the Data In (AWS S3 → Snowflake) 
**The problem**: Reports exist as files somewhere. We need them inside a database we can 
actually query. 
**The reasoning**: Rather than loading files directly into the "real" tables, best practice is to load 
them into a separate untouched **RAW** copy first. Why? Because if something goes wrong later in 
cleaning, you can always go back to the original — like keeping a photocopy of a document before 
you start editing it. 
**What actually happened**: S3 held the files → Snowflake connected to S3 through something 
called an "external stage" → a command called `COPY INTO` pulled the file in. 
**What went wrong and why it makes sense**: The connection between Snowflake and S3 initially 
failed. Think of it like a keycard that used to open a door but the building changed the lock — the 
permission/security certificate (called an "External ID") was outdated after the connection was 
rebuilt. Once it was refreshed with the current credentials, the connection worked. 
**How we knew it worked**: simple checks — does the file show up? Does the row count in 
Snowflake match the row count in the original file exactly, with zero errors? Both yes. --- 
## Stage 2: Cleaning the Mess (Dataiku) 
**The problem, in plain terms**: Real-world data is never clean. Someone typed "USA" here and 
"usa" there. Some dates got entered wrong. Some values are just missing. And — this is the big one 
— **the raw file had about 9,000 rows, but it turned out to be the same ~82 real reports repeated 
hundreds of times.** 
**Why that duplication happened / why it matters**: Think of it like a spreadsheet where someone 
accidentally copy-pasted the same block of data over and over while merging files. If we trained a 
model on 9,000 rows without noticing this, the model would essentially be memorizing the same 82 
stories repeated — a completely misleading picture. So the very first and most important cleaning 
step was **deduplication**: first remove exact duplicate rows, then do a second smarter pass that 
catches "same report but written slightly differently." This is why the final dataset is small (82 
records) — it's not a mistake, it's what remained after removing fake repetition. 
**Then, the smaller fixes, and why each one matters**: - **Inconsistent casing/spelling** ("Usa" vs "USA") → if left alone, the model would think these are 
two *different* countries. Fixed by standardizing text. - **Broken dates** → a few dates parsed as the year "0002" — clearly a formatting bug, not a real 
date. If you left that in, any time-based analysis (like "how many reports per month") would be 
corrupted. Caught by a simple sanity rule ("flag anything before year 2020") and corrected. - **Missing values** (e.g., missing dose amount) → instead of deleting those rows (we can't afford 
to lose data with only 82 records) or making up a number silently, we filled it with a reasonable 
estimate (the median) **and** kept a separate flag column saying "this value was originally 
missing." That way, the model can still learn that "missingness itself" might be meaningful, instead of 
the gap being hidden. 
**The reasoning behind flagging instead of just imputing silently**: If a doctor didn't record a 
patient's age, that itself might correlate with rushed/incomplete reporting, which could correlate 
with report quality. Hiding that by just filling in a number throws away a genuinely useful signal. --- 
## Stage 3: Organizing the Data Properly (Star Schema) 
**The problem**: Once clean, all this info (drug name, patient age, country, dose, etc.) is still just 
one flat pile. For a warehouse to be efficient and for features to be reusable across all 4 ML tasks, it 
needs proper structure. 
**The reasoning (in plain terms)**: Instead of one giant table repeating "USA" a hundred times in a 
column, you make a separate small "Country" table, a separate "Drug" table, a separate "Patient" 
table, etc., and the main table just *points to* them. This is called a **star schema** — one central 
table (`FACT_AE_REPORT`, the actual events) surrounded by lookup tables (dimensions). It's the 
standard way real companies structure data warehouses, because it's efficient and avoids repeating 
the same text everywhere. 
**How we checked it was actually correct**: - Does the number of rows in the final structured table still equal 82 (nothing lost or duplicated in the 
process)? Yes. 
- Does every row correctly link to a real drug/patient/country (no "orphan" rows pointing to 
nothing)? Initially no — 41 rows failed to link to the patient table because a missing-value flag 
column had blank values instead of a proper 0/1. Fixed by explicitly converting blanks to 0 before 
joining. - There was also a subtler bug: Snowflake is case-sensitive about column names in certain situations, 
and one tool (Snowpark) had quietly created a column named `"month"` in lowercase, which didn't 
match the `MONTH` column expected everywhere else — like two people using different keys that 
look identical but don't fit the same lock. Fixed by explicitly renaming to a consistent format. --- 
## Stage 4: The Machine Learning — 4 Separate Questions 
Before diving into each task, here's the **one rule that applies to all four**, and *why* it's the most 
important decision in the whole project: 
> **With only 82 records, you cannot trust a single train/test split, and you cannot trust a good
looking score without comparing it to a "dumb" baseline.** 
Why? If you randomly split 82 rows once, you might get lucky or unlucky purely by chance — the 
score would bounce around depending on which rows landed where. So instead, the models are 
tested using **cross-validation**: split the data into 5 chunks, train on 4, test on the 1 left out, 
repeat 5 times so every row gets tested exactly once, then average the results. This gives a much 
more stable, trustworthy number. 
And separately — a model's accuracy number means *nothing* on its own. If 51% of reports are 
"Non-serious," a model that just always guesses "Non-serious" gets 51% accuracy for free, without 
learning anything. So every single task is compared against that kind of "dumb guesser" baseline. 
**A model is only meaningful if it beats this baseline.** 
Also: features that are "answers in disguise" were deliberately removed. For example, `OUTCOME` 
(what actually happened to the patient) was excluded from predicting `SERIOUSNESS`, because 
outcome is basically determined *after* seriousness is already known — using it would be like 
predicting the exam result using the exam result. This is called **leakage**, and avoiding it was a 
conscious design decision, not an accident. 
With that foundation, here's what each task actually asked and found: 
### Task A: "Is this event Serious or Not?" 
**The question**: Given a report's details (age, drug, dose, country, etc.), can we predict whether 
it's Serious? 
**The reasoning for model choice**: Try a simple straight-line model (Logistic Regression) first, then 
two more flexible "tree-based" models (Random Forest, XGBoost) that can pick up on complicated, 
non-obvious patterns. Comparing all three tells you whether the true relationship in the data is 
simple (linear) or complex (non-linear). 
**What we found, and why it makes sense**: The simple model actually did *worse than guessing 
randomly*. That's a strong hint the relationship isn't a straight line. The tree-based models did 
modestly *better* than the "dumb guesser" baseline (baseline ≈ 51%, tree models ≈ 53–55%) — so 
there is *some* real, weak signal, but it's small. That's an honest, believable result for only 82 
records — not a failure, just a limitation of data size. 
### Task B: "What's the likely Cause (Certain / Probable / Possible / Unlikely / Unrelated)?" 
**The question**: Instead of a yes/no answer, predict one of 5 causality categories. 
**Why this is much harder, in plain terms**: Some categories have way fewer examples than others 
— only 9 reports were labeled "Certain" out of 82, but 31 were "Possible." Imagine trying to learn 
what a rare animal looks like from only 9 photos while you have 31 photos of a common one — the 
model will naturally lean toward guessing the common one every time, because doing so is "safe" 
and lowers its average error. To fight this, the training was told to pay *extra attention* to rare 
classes (a technique called class balancing) — but even with that help, the sample size was just too 
small. 
**What we found**: The models mostly kept guessing the majority class ("Possible") regardless of 
the true answer, and almost never correctly identified "Certain" — the rarest but most clinically 
important label. **Honest verdict**: this task isn't reliable enough yet to automate — and that's a 
legitimate, useful conclusion, because it tells a real team "don't trust an AI to auto-assign causality 
with this little data." 
### Task C: "How many reports will we get next month, per drug?" 
**The question**: Forecast future report *volume*, useful for planning staff/workload. 
**A problem discovered mid-way, and the reasoning for fixing it**: When we first looked at the 
data at the most detailed level (drug × specific batch × specific week), almost every single group had 
a report count of exactly 1 — meaning there was basically no variation to learn from at all. It's like 
trying to predict tomorrow's weather from a dataset where every single day so far said "sunny" — 
there's nothing to model. **The fix**: zoom out to a coarser view — drug × month instead of drug × 
batch × week — so there was at least *some* variation (counts ranging 1 to 4) to actually learn a 
pattern from. 
**What we found, and why this is actually the most instructive result in the project**: Every model 
we tried — a count-specific statistical model (Poisson Regression) and a tree-based model (XGBoost) 
— performed *worse* than simply guessing "the average count every time" (the baseline). That's a 
clear, honest signal: **there just isn't enough historical data yet to forecast reliably.** This isn't a 
coding mistake — it's exactly what the baseline comparison is designed to catch, and reporting it 
honestly (rather than hiding it) is the responsible thing to do. 
### Task D: "Did report volume suddenly spike this week, for any drug?" 
**The question**: Instead of forecasting far into the future, just watch recent weeks and flag if 
something looks unusual *right now* — like a smoke detector rather than a weather forecast. 
**A false start, and why it's a good story to tell**: The very first version flagged almost *every single 
week* as anomalous. That's obviously useless — if everything is "abnormal," the word loses 
meaning. Investigating why revealed the data had been artificially filled with zeros in a way that 
made the pattern look perfectly flat and then any tiny blip looked "abnormal" by comparison. 
**Fix**: re-aggregate the data properly so it has genuine, natural variation, not artificial flatness. 
After the fix, only 3 out of 22 weeks were flagged — a small, believable, useful number of alerts. 
**The most important insight in the whole project — worth memorizing**: A *statistical* spike 
(more reports than usual) is not automatically a *safety* concern. So after flagging the 3 unusual 
weeks, we went one step further and actually looked at what those specific reports said: - One flagged week (Immunova) — the extra reports were both harmless ("Non-serious," "Unrelated" 
cause). **Just statistical noise.** - Two flagged weeks (Vasculin, Oncozel) — the extra reports included ones marked "Serious" with 
"Certain" causality — meaning a doctor was confident the drug *did* cause a serious event. **These 
deserve real attention.** 
This shows the real value of the whole pipeline: **statistics tells you *where* to look; a 
human/clinical review tells you *whether it actually matters*.** The system doesn't replace expert 
judgment — it focuses it on the 3 weeks worth checking, instead of making someone scan every 
week by hand. --- 
## Stage 5: Saving the Results Back 
Once every model was validated with cross-validation (the trustworthy method), a final version of 
each model was trained one more time on *all* 82 records (no held-out data needed at this point, 
because trustworthiness was already proven), and its predictions were written back into Snowflake 
as new tables — so the results are usable by an analyst querying the warehouse, not stuck in a 
notebook. 
**A small technical wrinkle worth knowing**: when you ask a model "which features mattered 
most," you can't ask the *cross-validation* models that question, because cross-validation internally 
makes temporary *copies* of the model for each fold and never keeps the original one you can 
inspect. So a separate, fresh copy of each model was trained specifically to extract feature 
importance — a subtle but important distinction if someone asks "how did feature importance work 
if you were doing cross-validation?" --- 
## How We Made Sure Everything Was Actually Correct (Testing, in reasoning form) 
Three different *kinds* of checks were used, each catching a different kind of problem: 
1. **"Did the plumbing work?"** — row counts matching at every stage, no orphaned links 
between tables, no accidental duplicates. This catches *data engineering* bugs. 
2. **"Is there really a pattern here, or are we imagining one?"** — statistical tests run *before* 
modeling (chi-square, correlation tests) to check if any individual feature is obviously related to the 
target. This catches *false confidence* before wasting time building models on noise. 
3. **"Does the model's output actually make sense?"** — for example, if an anomaly detector flags 
100% of rows, something is definitely broken, because "everything is abnormal" is a logical 
contradiction. This catches *silent* bugs that don't throw an error but produce meaningless results. 
**28 out of 29 planned checks passed**, and 8 of them only passed *after* a real bug was found 
and fixed along the way — which is a good thing to say out loud, because it shows testing wasn't a 
checkbox at the end, it was built into every stage as things were built. 
**One item was left honestly unresolved** (whether one particular output table was 100% 
confirmed as saved) — rather than mark it "passed" without being sure, it was disclosed as still 
needing a final check. That honesty is worth highlighting if anyone asks about limitations. --- 
## If Someone Asks... (reasoning-based answers, not scripted lines) 
**"Why is your dataset only 82 rows — isn't that too small?"** 
Because the raw file looked like it had 9,000 rows, but that was almost entirely duplication — the 
same handful of real reports copied many times. Once genuinely deduplicated, 82 unique reports 
remained. Rather than pretend the bigger number was real data, the honest, smaller number was 
used — which is exactly why several of the ML results are modest. 
**"Why do some of your models perform worse than a dumb baseline?"** 
Because that's what actually happened when tested properly, and reporting it honestly is more 
valuable than hiding it. The baseline exists specifically to catch this — if I only reported the model's 
raw score without comparing it to a baseline, you'd have no way to know if 0.58 RMSE is actually 
good or bad. Comparing revealed it wasn't good, on this amount of data. 
**"How do you know your model isn't just memorizing the data?"** 
Two safeguards: cross-validation (each row is only ever *tested* by a model that never trained on it), 
and out-of-fold prediction for the multi-class task specifically. We also deliberately showed what 
happens *without* this safeguard (Task B's full-data fit gave a fake 100% accuracy) specifically to 
prove why the safeguard matters. 
**"What was the hardest part of this project?"** 
Realistically: dealing with a dataset too small for 3 of the 4 ML tasks to show strong results, while still 
being scientifically honest about that instead of tuning the numbers to look better. The anomaly 
detection task (Task D) was the one place the small dataset didn't hurt as much, because it only 
needs to notice deviation from a recent local pattern, not learn a complex relationship across many 
features. 
**"What would you change with more time/data?"** 
Get a bigger, multi-year dataset (the root cause of most weak results), add more contextual features 
(e.g., drug interaction data, patient medical history), and connect Task D's flags into a live dashboard 
instead of a one-time notebook run. --- 
## Words You Should Be Able to Explain in One Sentence - **Baseline** — the "dumb guess" score you must beat for a model to mean anything. - **Cross-validation** — testing a model multiple times on different slices of data so one 
lucky/unlucky split doesn't fool you. - **Leakage** — accidentally letting the model "see the answer" through a feature it shouldn't have 
access to. - **Star schema** — one central table of events, surrounded by small reusable lookup tables (drug, 
country, patient, etc.). - **Class imbalance** — when one category has far more examples than another, making models 
lazily favor the common one. - **Anomaly detection** — flagging *unusual* recent behavior, as opposed to forecasting, which 
predicts *future* behavior. - **Grain** — what one row in a table actually represents (e.g., "one row = one report + one 
adverse event"). 
