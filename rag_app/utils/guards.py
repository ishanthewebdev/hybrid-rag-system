def is_context_valid(context, min_len=50):
    return context and len(context.strip()) >= min_len
