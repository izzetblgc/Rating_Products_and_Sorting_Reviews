import pandas as pd
import math
import datetime as dt
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.shape

## Amazonda 2014 yılında elektronik kategorisinde en çok yorum alan ürünü inceliyoruz ##

### Ürünün ortalama ratingini güncel puanlamalara göre yeniden hesaplamak bize daha iyi sonuçlar verecektir

df["overall"].value_counts()

df["overall"].mean() #Ortalama rating 4.58758

df.info()

#reviewTime'ı datetime değişkenine dönüştürelim
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

df["reviewTime"].max() #veri setindeki son tarih

current_date = pd.to_datetime('2014-12-08 0:0:0')

## Ürünün ortalama ratingi 4.58758. Bu ratingi güncel puanlamalara göre değerlendirelim

df["days"] = (current_date - df['reviewTime']).dt.days

df["days"].max()

# Ağırlıkları belirlemek için günleri gözlemleyelim
pd.cut(df["days"], [0,100,200,300,400,600,750,1000,1065]).value_counts()

## Güncel puanlamalara göre ratingi ağırlıklandırma ##

df.loc[df["days"] <=100 , "overall"].mean() * 30 / 100 + \
df.loc[(df["days"] > 100) & (df["days"] <= 300), "overall"].mean() * 28 / 100 + \
df.loc[(df["days"] > 300) & (df["days"] <= 600), "overall"].mean() * 22 / 100 + \
df.loc[(df["days"] > 600), "overall"].mean() * 20 / 100

# Güncel puanlara göre ağırlıklandırdığımızda ortalama rating 4.63766 olarak
# hesaplanmaktadır


############ Ürün yorumlarının sıralanması ############

## Ürün sayfasında hangi yorumların üstlerde olacağını yorumlara verilen oyları baz alarak sıralama ##

df.head()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


## Yorumları sıralamak için yukardaki üç yöntem de kullanılabilir. Fakat ilk iki yöntem yanıltıcı olabilir.

## Doğru sonuç için oran yapmak şarttır o yuzden score_up_down_diff fonksiyonu işlevsiz kalacaktır.

## score_average_rating yönteminde yapılan oranlamada da toplam değerlendirme sayısı göz ardı edilebilir.
## Örneğin (2,0) ve (100,1) değerlendirmesine sahip 2 yorumda 100 up 1 down alan yorum social proof'a daha uygundur
## fakat bu yöntemde ilk örneği üste çıkaracaktır. score_average_rating fonksiyonu da bu yüzden işlevini yitirmektedir.


## Bunun için wilson lower bound yöntemini kullanacağız. Bernoulli yöntemini baz alan yöntem bize olasılıksal, yüzde 5 yanılma payıyla
## ve objektif bir sıralama yapma imkanı sunmaktadır.

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

reviews = df.sort_values("wilson_lower_bound", ascending=False).head(25)

## Ürün sayfasında en yukarda görünecek 25 yorum

print(reviews)




