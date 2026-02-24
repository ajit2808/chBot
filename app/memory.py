class ConversationMemory:
    def __init__(self):
        self.history = []

    def add(self, question, answer):
        self.history.append({"q": question, "a": answer})

    def get_context(self):
        context = ""
        for turn in self.history[-5:]:
            context += f"Q: {turn['q']}\nA: {turn['a']}\n"
        return context