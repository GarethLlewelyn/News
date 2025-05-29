import re
from datetime import datetime, timedelta

# --- Configuration for Relevance Classification ---

# Keywords and patterns for short-term relevance
# Order can matter if some phrases are subsets of others (though regex handles some of this)
SHORT_TERM_PATTERNS = [
    # Specific timeframes
    r"today", r"tonight", r"tomorrow",
    r"next\s+(few\s+)?(hour|day|week)s?",
    r"coming\s+(hour|day|week)s?",
    r"this\s+(morning|afternoon|evening|week)",
    r"in\s+the\s+coming\s+days?",
    r"over\s+the\s+next\s+\d+\s+(hour|day|week)s?",
    r"short[-\s]?term",
    r"immediate(ly)?",

    # Event-related words often implying recency or near future
    r"announced", r"launches", r"releases", r"introduces",
    r"reports\s+(earnings|results)", # e.g., "reports earnings today"
    r"earnings\s+call",
    r"webcast", r"conference\s+call",
    r"(just|now)\s+published",
    r"update(d)?",
    r"breaking(?!\s+ground)", # Avoid "breaking ground" for long-term projects
    r"alert",

    # Market movements, often short-term impact
    r"stock\s+(soars|plunges|drops|gains|falls|rises|jumps)",
    r"share\s+price\s+(up|down)",
    r"market\s+reaction"
]
SHORT_TERM_SCORE = 0.9 # Default score if a short-term keyword is found

# Keywords and patterns for long-term relevance
LONG_TERM_PATTERNS = [
    # Specific future years/dates
    r"(by|in|through|fiscal\s+year)\s+(20[2-9]\d)", # e.g., "by 2025", "fiscal year 2026"
    r"(FY[2-9]\d)", # e.g., FY25
    r"next\s+(few\s+)?(month|quarter|year)s?",
    r"coming\s+(month|quarter|year)s?",
    r"this\s+(quarter|year)", # Can be ambiguous, but often forward-looking in context
    r"over\s+the\s+next\s+\d+\s+(month|quarter|year)s?",
    r"long[-\s]?term",
    r"strategic\s+(plan|roadmap|outlook|initiative|vision)",
    r"outlook\s+for",
    r"forecast(s|ed)?",
    r"guidance\s+for",
    r"\d+[-\s]?year\s+plan", # e.g., 5-year plan
    r"future\s+(growth|development|prospects|plans)",
    r"expansion\s+plans?",
    r"long[-\s]?range",
    r"sustainable", r"sustainability",
    r"pipeline\s+(of|for)\s+(products|drugs|projects)", # often longer term development
    r"research\s+and\s+development", r"R&D",
    r"multi[-\s]?year",
    r"breaking\s+ground", # Start of a long project
    r"phased\s+rollout"
]
LONG_TERM_SCORE = 0.85 # Default score if a long-term keyword is found

# Keywords that might indicate non-relevance or lower relevance for financial impact
NON_RELEVANT_PATTERNS = [
    r"profile\s+of", r"biography",
    r"history\s+of",
    r"anniversary",
    r"retires", r"steps\s+down", # Often less direct immediate impact than new appointment
    r"appoints\s+new\s+(ceo|cfo|cto|coo|director|manager)", # This one is tricky, could be ST. Prioritize ST if both.
    r"obituary",
    r"feel[-\s]?good\s+story",
    r"human\s+interest",
    r"lifestyle", r"travel",
    r"opinion", r"editorial", r"blog\s+post", # Depending on source, might be less factual news
    r"survey\s+results", r"poll\s+data", # Unless directly market-moving
    r"weekly\s+roundup", r"summary\s+of\s+the\s+week"
]
NON_RELEVANT_SCORE_MODIFIER = -0.3 # Deduct from score if non-relevant terms found

# --- Relevance Classifier Class ---

class RelevanceClassifier:
    def __init__(self):
        # Compile regex patterns for efficiency
        self.short_term_regex = [re.compile(p, re.IGNORECASE) for p in SHORT_TERM_PATTERNS]
        self.long_term_regex = [re.compile(p, re.IGNORECASE) for p in LONG_TERM_PATTERNS]
        self.non_relevant_regex = [re.compile(p, re.IGNORECASE) for p in NON_RELEVANT_PATTERNS]
        print("RelevanceClassifier initialized with compiled regex patterns.")

    def classify_relevance(self, article_text, article_title=""):
        """Classifies article relevance based on keywords in title and text.
        Returns a dictionary: {"relevance_label": "short_term"/"long_term"/"non_relevant", 
                                "relevance_score": 0.0-1.0}
        """
        if not article_text and not article_title:
            return {"relevance_label": "non_relevant", "relevance_score": 0.0}

        text_to_search = (article_title.lower() + " " + article_text.lower()).strip()
        
        is_short_term = False
        is_long_term = False
        is_non_relevant_cue = False

        # Check for short-term cues
        for pattern_regex in self.short_term_regex:
            if pattern_regex.search(text_to_search):
                is_short_term = True
                break
        
        # Check for long-term cues
        for pattern_regex in self.long_term_regex:
            if pattern_regex.search(text_to_search):
                is_long_term = True
                break

        # Check for non-relevant cues
        for pattern_regex in self.non_relevant_regex:
            if pattern_regex.search(text_to_search):
                is_non_relevant_cue = True
                break

        # Determine label and score based on findings
        # Priority: Short-term > Long-term > Non-Relevant (if conflicting positive cues)
        # Non-relevant cues primarily act as a score dampener if other cues are weak or absent.

        current_label = "non_relevant" # Default
        current_score = 0.5 # Base score for neutral, can be adjusted

        if is_short_term and is_long_term:
            # If both, often short-term action within long-term context. Prioritize short-term.
            # Or could be "ambiguous" or require more sophisticated tie-breaking.
            # For now, let's lean towards short-term if both are strongly indicated.
            current_label = "short_term"
            current_score = max(SHORT_TERM_SCORE, LONG_TERM_SCORE) - 0.1 # Slightly penalize ambiguity
        elif is_short_term:
            current_label = "short_term"
            current_score = SHORT_TERM_SCORE
        elif is_long_term:
            current_label = "long_term"
            current_score = LONG_TERM_SCORE
        else: # Neither explicitly short nor long term detected by primary cues
            current_label = "non_relevant" 
            current_score = 0.3 # Low score if no strong temporal cues

        if is_non_relevant_cue:
            # If primary label is already non_relevant, reduce score further or cap it low.
            if current_label == "non_relevant":
                current_score = min(current_score, 0.2)
            else:
                # If it had a short/long term label, a non-relevant cue might reduce confidence
                current_score += NON_RELEVANT_SCORE_MODIFIER 
        
        # Ensure score is within 0.0 - 1.0
        final_score = max(0.0, min(1.0, round(current_score, 2)))

        # If score is very low, it might be better to classify as non_relevant regardless of initial ST/LT tag
        if final_score < 0.4 and current_label != "non_relevant":
            # print(f"Original label {current_label} with score {final_score} re-classified to non_relevant due to low score.")
            current_label = "non_relevant"

        return {"relevance_label": current_label, "relevance_score": final_score}

# --- Example Usage (for testing this module independently) ---
if __name__ == "__main__":
    classifier = RelevanceClassifier()

    test_articles = [
        {"title": "Stock Soars After Earnings Announcement Today", "text": "Company X reported record profits, stock price is up."}, # ST
        {"title": "Company Unveils 5-Year Strategic Roadmap for 2028", "text": "The CEO outlined a long-term vision."}, # LT
        {"title": "Market Update: Next Week's Key Events", "text": "Analysts are watching for next week's jobs report."}, # ST 
        {"title": "The History of Company Y", "text": "Founded in 1950, it has a rich history."}, # NR
        {"title": "CEO Announces Retirement, Effective Next Year", "text": "John Doe will step down in 2025."}, # LT + NR cue
        {"title": "Quarterly Earnings Call Scheduled for Tomorrow", "text": "Discussion of Q2 results and outlook for next quarter."}, # ST (+ LT cue with "next quarter")
        {"title": "New Product Launching This Afternoon", "text": "The much-anticipated gadget will be available today."}, # ST
        {"title": "Future Growth Hinges on R&D Pipeline Maturing by 2026", "text": "Long-term investments in research."}, # LT
        {"title": "An Opinion on Market Trends", "text": "This blog post discusses various possibilities."}, # NR
        {"title": "Breaking News: Immediate Impact Expected", "text": "A major event just occurred."}, # ST
        {"title": "Company to break ground on new factory next month, operational in 2027", "text": "This is a multi-year project."} # LT (both cues)
    ]

    print("\n--- Testing Relevance Classifier ---")
    for i, article_data in enumerate(test_articles):
        title = article_data["title"]
        text = article_data["text"]
        result = classifier.classify_relevance(text, title)
        print(f"\nArticle {i+1}: "
              f"Title: '{title}'\n"
              f"  -> Label: {result['relevance_label']}, Score: {result['relevance_score']}") 