from nltk.data import normalize_resource_url
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from numpy import vectorize  #for array
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
stop_words = set(stopwords.words('english'))

def cleanTag(rawhtml):
        cleanr=re.compile('<.*?>')
        cleantext=re.sub(cleanr, '', rawhtml) #removes html tags
        res = re.sub(r'[^\w\s]', '', cleantext) #removes punct
        result = re.sub(r"\d+", "", res) #removes numbers
        return result
def stem_tokens(tokens, PSstemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(PSstemmer.stem(item))
    return stemmed

sentence_data = "sentiment he will be to review One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only ""has got all the polari"" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams"
s = cleanTag(sentence_data)
ps = PorterStemmer()
token = word_tokenize(s)
tokens_without_sw = [word for word in token if not word in stopwords.words()]
stemmedafter = stem_tokens(tokens_without_sw,ps)
print(tokens_without_sw)



            
            