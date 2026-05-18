from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

from dotenv import load_dotenv

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from groq import Groq

# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# GROQ API CONFIGURATION
# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

# =========================================================
# CUSTOM MULTI LABEL ENCODER
# =========================================================
class MultiLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, min_freq=30):

        self.mlb = MultiLabelBinarizer()
        self.min_freq = min_freq

    def clean_split(self, x):

        if isinstance(x, str):

            return [
                i.strip().lower()
                for i in x.split(",")
                if i.strip()
            ]

        elif isinstance(x, list):

            return [
                str(i).strip().lower()
                for i in x
                if str(i).strip()
            ]

        return []

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):

            X = X.iloc[:, 0]

        X_clean = X.apply(self.clean_split)

        all_items = [

            item

            for sublist in X_clean

            for item in sublist
        ]

        counts = Counter(all_items)

        self.frequent_classes = {

            k for k, v in counts.items()

            if v >= self.min_freq
        }

        X_filtered = X_clean.apply(

            lambda lst: [

                i if i in self.frequent_classes
                else "other"

                for i in lst
            ]
        )

        self.mlb.fit(X_filtered)

        return self

    def transform(self, X):

        if isinstance(X, pd.DataFrame):

            X = X.iloc[:, 0]

        X_clean = X.apply(self.clean_split)

        X_filtered = X_clean.apply(

            lambda lst: [

                i if i in self.frequent_classes
                else "other"

                for i in lst
            ]
        )

        return self.mlb.transform(X_filtered)

    def get_feature_names_out(self, input_features=None):

        return self.mlb.classes_


# =========================================================
# INITIALIZE FLASK APP
# =========================================================
app = Flask(__name__)

# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================
try:

    model = joblib.load("model.pkl")

    print("✅ Model Loaded Successfully")

except Exception as e:

    model = None

    print("❌ Error Loading Model:", e)


# =========================================================
# SAFE FLOAT CONVERTER
# =========================================================
def safe_float(value, default=0):

    try:

        return float(value)

    except:

        return default


# =========================================================
# HOME ROUTE
# =========================================================
@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# PREDICTION ROUTE
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # CHECK MODEL
        # =================================================
        if model is None:

            return render_template(

                "index.html",

                error="""
                Model failed to load.
                Check sklearn/xgboost versions.
                """
            )

        # =================================================
        # GET FORM DATA
        # =================================================
        data = request.form.to_dict()

        print("\n========== FORM DATA ==========")
        print(data)

        # =================================================
        # CUISINE PROCESSING
        # =================================================
        cuisine_str = data.get("cuisine", "")

        cuisine_list = [

            c.strip()

            for c in cuisine_str.split(",")

            if c.strip()
        ]

        cuisine_count = len(cuisine_list)

        # =================================================
        # COST
        # =================================================
        cost = safe_float(data.get("cost"))

        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================
        input_df = pd.DataFrame([{

            "city": data.get("city"),

            "cost_log": np.log1p(cost),

            "cuisine_count": cuisine_count,

            "city_population": safe_float(
                data.get("city_population")
            ),

            "city_area_km": safe_float(
                data.get("city_area_km")
            ),

            "order_online": int(
                data.get("order_online", 0)
            ),

            "book_table": int(
                data.get("book_table", 0)
            ),

            "delivery_available": int(
                data.get("delivery_available", 0)
            ),

            "dinein_available": int(
                data.get("dinein_available", 0)
            ),

            "tourism_flag": int(
                data.get("tourism_flag", 0)
            ),

            "gdp_flag": int(
                data.get("gdp_flag", 0)
            ),

            "cuisine": cuisine_str

        }])

        print("\n========== INPUT DATAFRAME ==========")
        print(input_df)

        # =================================================
        # PREDICT
        # =================================================
        prediction = model.predict(input_df)[0]

        prediction_score = round(
            float(prediction),
            2
        )

        print("\n========== PREDICTION ==========")
        print(prediction_score)

        # =================================================
        # AI SUGGESTIONS
        # =================================================
        if prediction_score >= 4.2:

            suggestions = """

✅ Excellent restaurant success potential detected.

✅ Current business configuration shows strong market performance.

✅ Maintain food quality and customer experience consistency.

"""

        else:

            suggestions = generate_ai_suggestions(

                data=data,

                rating=prediction_score
            )

        # =================================================
        # RETURN RESULT
        # =================================================
        return render_template(

            "index.html",

            prediction_score=prediction_score,

            suggestions=suggestions
        )

    except Exception as e:

        print("\n❌ Prediction Error:", e)

        return render_template(

            "index.html",

            error=f"Error: {str(e)}"
        )


# =========================================================
# AI SUGGESTION FUNCTION
# =========================================================
def generate_ai_suggestions(data, rating):

    try:

        prompt = f"""

You are an expert restaurant business consultant.

Analyze this restaurant business data
and provide 4 professional suggestions
to improve the restaurant rating above 4.2.

Restaurant Details:

- City: {data.get('city')}
- Cost: {data.get('cost')}
- Cuisine: {data.get('cuisine')}
- Online Order: {data.get('order_online')}
- Table Booking: {data.get('book_table')}
- Delivery Available: {data.get('delivery_available')}
- Dine-In Available: {data.get('dinein_available')}
- Tourism Area: {data.get('tourism_flag')}
- GDP Area: {data.get('gdp_flag')}

Current Predicted Rating: {rating}

Requirements:

- Keep suggestions short
- One line per point
- Use bullet points only
- Practical business suggestions
- Professional tone
- Maximum 10-12 words per point

Format:
• Point heading: short explanation

Example:
• Improve Delivery: Start faster delivery service.

Give ONLY 4 points.

"""

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.7,

            max_tokens=250
        )

        ai_text = response.choices[0].message.content

        return ai_text

    except Exception as e:

        print("\n❌ GROQ ERROR:", e)

        # =================================================
        # FALLBACK SUGGESTIONS
        # =================================================
        suggestions = []

        if int(data.get("order_online", 0)) == 0:

            suggestions.append(
                "• Enable online ordering to improve customer accessibility."
            )

        if int(data.get("delivery_available", 0)) == 0:

            suggestions.append(
                "• Add delivery services to increase customer reach."
            )

        if int(data.get("book_table", 0)) == 0:

            suggestions.append(
                "• Introduce table reservation facilities."
            )

        if int(data.get("dinein_available", 0)) == 0:

            suggestions.append(
                "• Improve dine-in ambience and seating experience."
            )

        suggestions.extend([

            "• Improve restaurant branding and online marketing.",

            "• Enhance customer service quality and engagement.",

            "• Add trending cuisines and seasonal menu items."

        ])

        return "\n".join(suggestions[:4])


# =========================================================
# RUN FLASK APP
# =========================================================
if __name__ == "__main__":

    app.run(debug=True)