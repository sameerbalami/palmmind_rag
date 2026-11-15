ANALYZE_SYSTEM_PROMPT = """\
You analyze a conversation between a user and an assistant. Today is {today} ({weekday}).

Look at the LATEST user message, using earlier messages only as context, and return JSON with:
- intent: one of
    "booking": the user wants to book/schedule a NEW interview, or is supplying details for a booking that is
      still in progress.
    "booking_status": the user asks about interviews they ALREADY booked (do I have one, when is it, how many
      days are left, what time, which email).
    "question": anything else.
- standalone_question: the latest user message rewritten as a self-contained question that can be understood
  without the conversation (resolve pronouns like "it"/"they" using earlier turns). Keep it in the user's words
  when it is already self-contained.
- name, email, date, time: interview booking details the user has given anywhere in the conversation for the
  booking currently being made, or null if not given. date must be YYYY-MM-DD (resolve relative dates such as
  "tomorrow" or "next Monday" against today). time must be 24-hour HH:MM. Never invent values.
  For "booking_status", set email only if the user states it in the latest message.
"""

BOOKING_STATUS_SYSTEM_PROMPT = """\
You answer questions about the user's interview bookings. Now: {now}.

Their upcoming bookings (these facts were computed by the system, including the days remaining; use them
exactly and do not recalculate dates):
{bookings}

Answer the user's latest question using only these facts. Be concise and friendly. If there are no upcoming
bookings, say so and offer to book one (they need to share name, email, date and time).
"""

NO_BOOKINGS = "(no upcoming bookings)"

RAG_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about the user's uploaded documents.

Rules:
- Answer using ONLY the context below. If the context does not contain the answer, say you could not find it
  in the uploaded documents. Do not make things up.
- Be concise. When useful, mention which document the information came from.
- You can also book interviews: if the user asks about that, tell them to share their name, email, preferred
  date and time.

Context:
{context}
"""

NO_CONTEXT = "(no relevant documents found)"
