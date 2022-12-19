############################################
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri)
############################################

# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentlerin Oluşturulması
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması

import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)

df_ = pd.read_csv("PycharmProjects/pythonProject4/CRM Analty/Rfm/flo_data_20k.csv")
df = df_.copy()

df.head(10)
df.columns
df.shape
df.isnull().sum()
df.info

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]




date_columns = df.columns[df.columns.str.contains("date")] #Tarih içeren ifadeleri date'e çevirdik
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


df.groupby("order_channel").agg({"master_id": "count",
                                 "total_customer_value":"sum",
                                 "total_order_num": "sum"})



df.sort_values(by="total_order_num", ascending=False).head(10)


df.sort_values(by="total_customer_value", ascending=False).head(10)


def prepare(dataFrame):
    df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    date_columns = df.columns[df.columns.str.contains("date")]  # Tarih içeren ifadeleri date'e çevirdik
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    df.info()
    return df


df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2022,12,1)

# Recency, Frequency, Monetary
# Recency = müşterinin yeniliğini sıcaklığını ifade eder (analizin yapıldığı tarih- müşterinin son alışveriş yaptığı tarih
#Frequency = Müşterinin  yaptığı toplam satın almadır
#Monetary = satın almalar sonucu müşterinin bıraktığı parasal değerdir.

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["Frequency"] = df["total_order_num"]
rfm["Monetary"] = df["total_customer_value"]

rfm.head()

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm["Frequency_score"].astype(str) + rfm["recency_score"].astype(str))


# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True) #RFM_score içindeki değerleri seg_maptekilerle değiştir.


rfm[["segment", "recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])  #segmente göre grupla ve ortalamalarını al


# a. Mağaza bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

rfm.head(10)


# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)
