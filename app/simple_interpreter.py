"""
Simple ASL Letter Recognition (No Sentence Generation)
======================================================
Just shows the ASL letters you sign - no complex sentence processing.
"""

def tokens_to_sentence(tokens: list[str]) -> str:
    """
    Simply join the ASL letters/tokens into a readable format.
    No grammar processing - just show what was signed.
    """
    if not tokens:
        return ""
    
    # Just join the tokens with spaces
    return " ".join(tokens)