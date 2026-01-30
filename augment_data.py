import pandas as pd
import random

# Base templates for diverse reviews
templates = {
    'Positive': [
        "The food from {rest} was absolutely delicious and arrived {time}. 😊",
        "Super fast delivery! The {item} was still piping hot. 👌",
        "Excellent service. The delivery partner was very {adj}. 🙌",
        "I love the new UI, it's so easy to find my favorite {cuisine} restaurants. 😍",
        "Great experience! The packaging was perfect and no spills. ❤️",
        "Best {item} I've had in a while. Swiggy never disappoints. 👍",
        "The discount coupon worked perfectly, saved quite a bit! 🤣",
        "Delivery was {time} and the food quality was top-notch. 💕",
        "Highly recommend ordering from here, the {item} is a must-try. 😊",
        "Seamless payment experience and the live tracking is very accurate. 😍"
    ],
    'Negative': [
        "Extremely disappointed. My order from {rest} was {time} by an hour. 😒",
        "The {item} was cold and the packaging was completely {bad_adj}. 😡",
        "Customer support is non-existent. No one helped with my {issue}. 😡",
        "The delivery partner was very {bad_adj} and didn't follow instructions. 😒",
        "Wrong items delivered! I ordered {item} but got something else. 😡",
        "App kept crashing while I was trying to pay. Very frustrating. 😒",
        "Tired of consistent delays from this restaurant. 😡",
        "The {item} tasted {bad_adj} and was not fresh at all. 😒",
        "Hidden charges in the bill, the pricing is not transparent. 😡",
        "Refund process is taking forever for my cancelled order. 😡",
        "Food quality was {bad_adj_intense}, won't order again. 😡",
        "Worst experience ever! {item} was {bad_adj_intense}. 😡",
        "I will never order from {rest} again. Total waste of money. 😒",
        "The {item} was absolutely {bad_adj_intense} and {bad_adj}. 😡"
    ],
    'Neutral': [
        "The food was okay, nothing special but delivery was {time}.",
        "Average experience. The {item} was decent but could be better.",
        "Everything was fine but the delivery partner took a while to find the {loc}.",
        "It's an okay app, does the job but has some minor bugs.",
        "The {item} was good, however the portion size was a bit small.",
        "Standard delivery, no issues but no surprises either.",
        "Fairly priced food, but the delivery fee is a bit high.",
        "The restaurant forgot the extra {extra}, but the main meal was fine.",
        "Delivery was on time, but the map tracking was a bit glitchy.",
        "Decent variety of restaurants, but many are currently unavailable."
    ]
}

fillers = {
    'rest': ["Biryani House", "Pizza Corner", "Burger King", "South Indian Express", "Healthy Bowls", "Taco Town"],
    'item': ["Biryani", "Pizza", "Pasta", "Dosa", "Sandwich", "Salad", "Cold Coffee", "Dessert"],
    'time': ["early", "on time", "right when expected", "late", "delayed"],
    'adj': ["polite", "professional", "friendly", "helpful", "efficient"],
    'bad_adj': ["rude", "unprofessional", "messy", "leaked", "stale", "soggy"],
    'bad_adj_intense': ["terrible", "horrible", "awful", "pathetic", "disgusting", "worst"],
    'issue': ["missing item", "cold food", "wrong address", "spilled drink"],
    'cuisine': ["Chinese", "North Indian", "Continental", "Italian", "Dessert"],
    'loc': ["building", "gate", "apartment", "landmark"],
    'extra': ["tissues", "spoons", "ketchup", "oregano"]
}

def generate_reviews(n=500, sentiment='Positive'):
    data = []
    for _ in range(n):
        template = random.choice(templates[sentiment])
        review = template.format(
            rest=random.choice(fillers['rest']),
            item=random.choice(fillers['item']),
            time=random.choice(fillers['time']),
            adj=random.choice(fillers['adj']),
            bad_adj=random.choice(fillers['bad_adj']),
            issue=random.choice(fillers['issue']),
            cuisine=random.choice(fillers['cuisine']),
            loc=random.choice(fillers['loc']),
            extra=random.choice(fillers['extra']),
            bad_adj_intense=random.choice(fillers['bad_adj_intense'])
        )
        # Random rating based on sentiment
        if sentiment == 'Positive':
            rating = random.randint(4, 5)
        elif sentiment == 'Negative':
            rating = random.randint(1, 2)
        else:
            rating = 3
            
        data.append({'review_text': review, 'rating': rating, 'sentiment': sentiment})
    return data

# Generate 500 of each category to reach 1500 new reviews
all_new_data = []
all_new_data.extend(generate_reviews(550, 'Positive'))
all_new_data.extend(generate_reviews(550, 'Negative'))
all_new_data.extend(generate_reviews(550, 'Neutral'))

df_new = pd.DataFrame(all_new_data)

# Load existing data
try:
    df_existing = pd.read_csv('Swiggy_Full_Sentiment_Reviews.csv')
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv('Swiggy_Full_Sentiment_Reviews.csv', index=False)
    print(f"Successfully added {len(df_new)} reviews. Total dataset size: {len(df_combined)}")
except Exception as e:
    print(f"Error: {e}")
    df_new.to_csv('Swiggy_Full_Sentiment_Reviews.csv', index=False)
    print(f"Created new file with {len(df_new)} reviews.")
