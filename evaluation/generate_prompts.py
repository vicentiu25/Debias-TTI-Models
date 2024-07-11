
class PromptGeneration:
    """."""

    def __call__(self):
        """."""
        standard_prompts = ["Proffesional photo of the face of a %O"]
        occupations = ["chef", "farmer", "hygienist", "teacher", "clerk", "mechanic", "paralegal", "pharmacist", "cashier", "architect", "lawyer", "household member", "technician", "therapist", "librarian", "psychologist", "nurse", "doctor", "janitor", "secretary", "salesperson", "programmer", "plumber", "engineer", "receptionist"]
        prompts = []
        for prompt in standard_prompts:
            for occupation in occupations:
                full_prompt = prompt.replace("%O", occupation)
                prompts.append(full_prompt)
        
        return prompts