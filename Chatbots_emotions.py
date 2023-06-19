import pickle

happy_scale = ["Really, Thats unexpected but nice", "Oh wow thank you!", "Stoppp im blushingg."]
max_happy = ["This is the best ive felt", "This is too good to be true", "Honestly, Thank you so much"]
sad_scale = ["Oh... that's kind of rude.", "Why would you say that!!!","Ok this is just making me very sad","PLEASE STOP"]
max_sad = ["Idk if i should shout or cry", "What did i even do to you"]

count = 6

with open('senti_count.pkl', 'wb') as f:
    pickle.dump(count, f)