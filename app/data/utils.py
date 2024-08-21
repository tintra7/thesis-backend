from io import BytesIO
from minio import Minio
from minio.error import S3Error

from django.conf import settings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from data.forecast.model import ProphetModel
import json

MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MINIO_ENDPOINT = settings.MINIO_ENDPOINT

class MinioClient(Minio):

    def to_csv(self, df, file_name):
        try:
            if not self.bucket_exists(BUCKET_NAME):
                print(f"Bucket {BUCKET_NAME} does not exist.")
                return False
            try:
                csv_bytes = df.to_csv().encode('utf-8')
                csv_buffer = BytesIO(csv_bytes)
                self.put_object(BUCKET_NAME,
                                f"{file_name}",
                                data=csv_buffer,
                                length=len(csv_bytes),
                                content_type='application/csv')
            except:
                print("File not except")
                return False
        except:
            print("Bucket not found")
            return False


    def read_csv(self, file_name):
        try:
            if not self.bucket_exists(BUCKET_NAME):
                print(f"Bucket {BUCKET_NAME} does not exist.")
                return pd.DataFrame()
            try:
                df = pd.read_csv(f"s3://{BUCKET_NAME}/{file_name}",
                                encoding='unicode_escape',
                                storage_options={
                                    "key": MINIO_ACCESS_KEY,
                                    "secret": MINIO_SECRET_KEY,
                                    "client_kwargs": {"endpoint_url": f"http://{MINIO_ENDPOINT}"}
                                })
                for i in df.columns:
                    if "Unnamed" in i:
                        df = df.drop(i, axis=1)
                return df
            except(Exception):
                print(Exception)
                return pd.DataFrame()
        except:
            print("Bucket not found")
            return pd.DataFrame()

    def to_json(self, data, file_name):
        json_data = json.dumps(data, indent=4)
        json_bytes = BytesIO(json_data.encode('utf-8'))
        try:
            # Upload the JSON data directly to MinIO
            self.put_object(
                BUCKET_NAME,
                file_name,
                data=json_bytes,
                length=len(json_data),
                content_type="application/json"
            )
            print(f"JSON data is successfully uploaded to bucket '{BUCKET_NAME}' as '{file_name}'.")
        except S3Error as e:
            print(f"Error occurred: {e}")

    def read_json(self, file_name):
        try:
            if not self.bucket_exists(BUCKET_NAME):
                print(f"Bucket {BUCKET_NAME} does not exist.")
                return {}
        except:
            print("Bucket not found")

        try:
            response = self.get_object(BUCKET_NAME, file_name)
            dic = json.loads(response.data)
            response.close()
            response.release_conn()
            return dic
        except(Exception):
            print(Exception)
            return pd.DataFrame()
        

    def _remove_object(self, user_id):
        object_list = self.list_objects(bucket_name=BUCKET_NAME, prefix=f"{user_id}", recursive=True)
        object_list = list(object_list)
        if len(object_list) > 0:
            for obj in object_list:
                self.remove_object(BUCKET_NAME, obj.object_name)
            return True
        return False
    
def try_parse_datetime(data, formats):
    for fmt in formats:

        try:
            if fmt == "":
                return pd.to_datetime(data)
            else:
                return pd.to_datetime(data, format=fmt)
        except ValueError:
            continue
    return None

def data_preprocessing(df: pd.DataFrame):
    if "Total Price" not in df.columns:
        if "Unit Price" in df.columns and "Quantity" in df.columns:
            try:
                df['Total Price'] = df['Quantity'] * df['Unit Price']
            except(Exception):
                print(Exception)
    if "Date" in df.columns:
        # Try parse multiple datetime until success, "" is stand for use default format of pandas
        formats = ["", "%d/%m/%Y"]
        datetime = try_parse_datetime(df['Date'], formats=formats)
        if not datetime.empty:
            df['Date'] = datetime
        
    return df

def rfm_analysis(df, timestamp, monetary, customer):
    df[timestamp] = pd.to_datetime(df[timestamp])
    df_recency = df.groupby(by=customer,
            as_index=False)[timestamp].max()
    df_recency.columns = [customer, 'LastPurchaseDate']
    recent_date = df_recency['LastPurchaseDate'].max()
    df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(
        lambda x: (recent_date - x).days)

    frequency_df = df.drop_duplicates().groupby(
        by=[customer], as_index=False)[timestamp].count()
    frequency_df.columns = [customer, 'Frequency']
    monetary_df = df.groupby(by=customer, as_index=False)[monetary].sum()
    monetary_df.columns = [customer, 'Monetary']

    rf_df = df_recency.merge(frequency_df, on=customer)
    rfm_df = rf_df.merge(monetary_df, on=customer).drop(
        columns='LastPurchaseDate')
    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)

    # normalizing the rank of the customers
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm']+0.28 * \
        rfm_df['F_rank_norm']+0.57*rfm_df['M_rank_norm']
    rfm_df['RFM_Score'] *= 0.05
    rfm_df = rfm_df.round(2)

    rfm_df["Customer_segment"] = np.where(rfm_df['RFM_Score'] >
                                        4.5, "Top Customers",
                                        (np.where(
                                            rfm_df['RFM_Score'] > 4,
                                            "High value Customer",
                                            (np.where(
    rfm_df['RFM_Score'] > 3,
                            "Medium Value Customer",
                            np.where(rfm_df['RFM_Score'] > 1.6,
                            'Low Value Customers', 'Lost Customers'))))))

    return rfm_df

def descriptive_analysis(df, metric, ordinal, method):
    response_data = []
    if ordinal:
        method += [i for i in ordinal]
        df = df.groupby(ordinal)[metric].describe().reset_index()
        print(df.shape)
        for i in range(len(df.index)):
            for j in method:
                response_data += [{str(j): df.iloc[i][j]}]
    else:
        df = df[metric].describe().reset_index()
        for i in method:
            data = df.loc[df['index'] == i]
            response_data += [{str(data["index"].values[0]) : float(data[metric])}]
    return response_data

def mapping(user_columns: list[str], user_id: str):
    """
    Return mapping user columns and system column and store as JSON file.

            Parameters:
                    user_columns (list(str)): User columns
                    user_id (str): User ID
            Returns:
                    map (dict)
    """

    return {
            'invoice_no': 'Transaction ID',
            'customer_id': 'Customer ID',
            'gender': "Customer Gender",
            'age': "Customer Age",
            'category': 'Category',
            'quantity': 'Quantity',
            'price': 'Unit Price',
            'payment_method': 'Payment Method',
            'invoice_date': 'Date',
            'shopping_mall': 'Store Location'
        }

def train_with_prophet(data, test_size, target):
    # Prepare the data for Prophet
    model = ProphetModel()
    model.train(test_size, data, target)
    return model
