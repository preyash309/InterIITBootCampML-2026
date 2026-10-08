"""Closed, versioned discourse ontologies. Labels do not generate meeting facts."""

EVENT_ONTOLOGY = (
    ("INFORMATION", "A factual update without a choice or promised future task."),
    ("QUESTION", "A request for an answer, not a proposal or assignment."),
    ("ANSWER", "An answer to an earlier question."),
    ("PROPOSAL", "An unaccepted option, suggestion or conditional future plan."),
    ("SUPPORT", "Agreement or endorsement without a confirmed choice."),
    ("OBJECTION", "A concern against an option without explicit rejection."),
    ("REJECTION", "An explicit refusal of an option."),
    ("DECISION", "An explicit confirmed choice, including acceptance of a prior proposal."),
    ("COMMITMENT", "A speaker explicitly promises their own future work."),
    ("TASK_ASSIGNMENT", "An explicit work assignment to an identified person or team."),
    ("BLOCKER", "An explicit obstacle preventing progress."),
    ("CLARIFICATION", "Disambiguates the meaning or scope of an earlier statement."),
    ("NONE", "None of these classes applies."),
    ("AMBIGUOUS", "Multiple classes remain plausible or evidence is insufficient."),
)
RELATION_ONTOLOGY = (
    ("SUPPORTS", "B endorses A without accepting it as a final choice."),
    ("ACCEPTS", "B explicitly accepts proposal A as a confirmed choice."),
    ("ANSWERS", "B answers question A."),
    ("CONTRADICTS", "B disputes the content of A."),
    ("REJECTS", "B explicitly rejects option A."),
    ("SUPERSEDES", "Later confirmed decision B replaces earlier confirmed decision A."),
    ("CLARIFIES", "B clarifies the scope or meaning of A."),
    ("ASSIGNS", "B assigns work arising from A."),
    ("RESULTS_IN", "A explicitly leads to B; temporal succession alone is insufficient."),
    ("NONE", "No supported relation."),
    ("AMBIGUOUS", "The relation is unclear."),
)
POLICIES = (
    "event_ontology_v1",
    "event_classification_v1",
    "event_relation_v1",
    "semantic_verification_v1",
    "coverage_v1",
)
HIGH_VALUE = ("DECISION", "TASK_ASSIGNMENT", "COMMITMENT", "BLOCKER")
