from transformers import pipeline  # type: ignore
import torch  # type: ignore


class NLPAnalyzer:
    """
    NLP Engine for:
    - Named Entity Recognition
    - Sentiment Analysis
    - Abuse / Toxicity Detection
    - Spam Detection
    - Emotion Detection
    - Text Summarization
    - Zero-Shot Classification
    """

    def __init__(self):
        """
        Load all models ONCE when class is initialized
        """

        # -------------------------------
        # NER
        # -------------------------------
        self.ner_model = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            grouped_entities=True
        )

        # -------------------------------
        # Sentiment
        # -------------------------------
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        # -------------------------------
        # Abuse / Toxicity
        # -------------------------------
        self.abuse_model = pipeline(
            "text-classification",
            model="unitary/toxic-bert"
        )

        # -------------------------------
        # Spam Detection
        # -------------------------------
        self.spam_model = pipeline(
            "text-classification",
            model="mrm8488/bert-tiny-finetuned-sms-spam-detection"
        )

        # -------------------------------
        # Emotion Detection
        # -------------------------------
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

        # -------------------------------
        # Summarization
        # -------------------------------
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )

        # -------------------------------
        # Zero-Shot Classification
        # -------------------------------
        self.zero_shot_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    # ===============================
    # NER
    # ===============================
    def ner_detection(self, text: str):
        with torch.no_grad():
            return self.ner_model(text)

    def ner_insights(self, text: str):
        raw_entities = self.ner_detection(text)

        insights = {
            "People": [],
            "Organizations": [],
            "Locations": [],
            "Misc": []
        }

        for ent in raw_entities:
            label = ent["entity_group"]
            word = ent["word"].replace("##", "").strip()

            if label == "PER":
                insights["People"].append(word)
            elif label == "ORG":
                insights["Organizations"].append(word)
            elif label in ["LOC", "GPE"]:
                insights["Locations"].append(word)
            else:
                insights["Misc"].append(word)

        return {k: v for k, v in insights.items() if v}

    def ner_summary(self, text: str):
        insights = self.ner_insights(text)
        summary = []

        if "People" in insights:
            summary.append(f"People mentioned: {', '.join(insights['People'])}")
        if "Organizations" in insights:
            summary.append(f"Organizations mentioned: {', '.join(insights['Organizations'])}")
        if "Locations" in insights:
            summary.append(f"Locations mentioned: {', '.join(insights['Locations'])}")

        return " | ".join(summary)

    # ===============================
    # SENTIMENT
    # ===============================
    def sentiment_analysis(self, text: str):
        with torch.no_grad():
            result = self.sentiment_model(text)[0]
            return {
                "sentiment": result["label"],
                "confidence": round(result["score"], 3)
            }

    # ===============================
    # ABUSE / TOXICITY
    # ===============================
    def abuse_detection(self, text: str, threshold=0.6):
        with torch.no_grad():
            result = self.abuse_model(text)[0]

            is_abusive = (
                result["label"].lower() == "toxic"
                and result["score"] >= threshold
            )

            return {
                "abusive": is_abusive,
                "label": result["label"],
                "confidence": round(result["score"], 3)
            }

    # ===============================
    # SPAM DETECTION
    # ===============================
    def spam_detection(self, text: str, threshold=0.6):
        with torch.no_grad():
            result = self.spam_model(text)[0]
    
        is_spam = (
            result["label"] == "LABEL_1"
            and result["score"] >= threshold
        )
    
        return {
            "spam": is_spam,
            "label": "SPAM" if result["label"] == "LABEL_1" else "HAM",
            "confidence": round(result["score"], 3)
        }

    # ===============================
    # EMOTION DETECTION
    # ===============================
    def emotion_detection(self, text: str):
        with torch.no_grad():
            results = self.emotion_model(text)[0]
            top_emotion = max(results, key=lambda x: x["score"])

            return {
                "emotion": top_emotion["label"],
                "confidence": round(top_emotion["score"], 3)
            }

    # ===============================
    # TEXT SUMMARIZATION
    # ===============================
    def summarize_text(self, text: str, max_length=130):
        with torch.no_grad():
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]

            return summary

    # ===============================
    # ZERO-SHOT CLASSIFICATION
    # ===============================
    def zero_shot_classification(self, text: str, labels: list):
        with torch.no_grad():
            result = self.zero_shot_model(text, candidate_labels=labels)

            return {
                "label": result["labels"][0],
                "confidence": round(result["scores"][0], 3)
            }


# # Singleton instance
nlp = NLPAnalyzer()

# response = nlp.spam_detection("Congratulations! You won $1,000,000")

# response.pop("label")

# print(response)

# # spam detection : {'spam': False, 'label': 'LABEL_0', 'confidence': 0.921}

print(nlp.spam_detection(
    "FREE entry in 2 a wkly comp to win FA Cup final tkts. "
    "Text FA to 87121 to receive entry question"
))
