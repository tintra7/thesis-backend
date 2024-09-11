from io import BytesIO
import re
from minio import Minio
from minio.error import S3Error

from django.conf import settings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from data.forecast.model import LSTMModel, ProphetModel, XGBoostModel
import json
from openai import RateLimitError, OpenAI
import time
from sqlalchemy import create_engine, text


MINIO_ACCESS_KEY = settings.MINIO_ACCESS_KEY
MINIO_SECRET_KEY = settings.MINIO_SECRET_KEY
BUCKET_NAME = settings.MINIO_BUCKET_NAME
MINIO_ENDPOINT = settings.MINIO_ENDPOINT
API_KEY = settings.OPENAI_API_KEY

INPUT_PRICE = 0.15/1000000
OUTPUT_PRICE = 0.6/1000000

openai_client = OpenAI(api_key=API_KEY)

class MinioClient(Minio):

    def to_csv(self, df, file_name) -> bool:
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
                return True
            except:
                print("File not except")
                return False
        except:
            print("Bucket not found")
            return False


    def read_csv(self, file_name) -> pd.DataFrame:
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

    def to_json(self, data: dict, file_name) -> bool:
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
            return True
        except S3Error as e:
            print(f"Error occurred: {e}")
            return False

    def read_json(self, file_name) -> dict:
        try:
            if not self.bucket_exists(BUCKET_NAME):
                print(f"Bucket {BUCKET_NAME} does not exist.")
                return {}
        except:
            print("Bucket not found")
            return {}

        try:
            response = self.get_object(BUCKET_NAME, file_name)
            dic = json.loads(response.data)
            response.close()
            response.release_conn()
            return dic
        except(Exception):
            print(Exception)
            return {}

    def _remove_object(self, user_id) -> bool:
        object_list = self.list_objects(bucket_name=BUCKET_NAME, prefix=f"{user_id}", recursive=True)
        object_list = list(object_list)
        if len(object_list) > 0:
            for obj in object_list:
                self.remove_object(BUCKET_NAME, obj.object_name)
            return True
        return False
    
def try_parse_datetime(data):
    formats = ["%d/%m/%Y", ""]
    for fmt in formats:
        try:
            if fmt == "":
                return pd.to_datetime(data)
            else:
                return pd.to_datetime(data, format=fmt)
        except ValueError:
            continue
    return None

def remove_symbol(s):
    trim = re.compile(r'[^\d.,]+')

    result = trim.sub('', s)
    return result

def data_preprocessing(df: pd.DataFrame):
    if "Unit Price" in df.columns and not is_numeric_dtype(df['Unit Price']):
        df['Unit Price'] = df['Unit Price'].apply(remove_symbol)
        df['Unit Price'] = pd.to_numeric(df['Unit Price'])

    if "Total Price" in df.columns and not is_numeric_dtype(df['Unit Price']):
        df['Total Price'] = df['Total Price'].apply(remove_symbol)
        df['Total Price'] = pd.to_numeric(df['Total Price'])   

    if "Total Price" not in df.columns:
        if "Unit Price" in df.columns and "Quantity" in df.columns:
            try:
                df['Total Price'] = df['Quantity'] * df['Unit Price']
            except(Exception):
                print(Exception)
    if "Date" in df.columns:
        # Try parse multiple datetime until success, "" is stand for use default format of pandas
        datetime = try_parse_datetime(df['Date'])
        if not datetime.empty:
            df['Date'] = datetime
    df = df.dropna(axis=0)
    return df

score_dict = {'Champions': [515, 524, 525, 533, 534, 535, 542, 543, 544, 545, 551, 552, 553, 554, 555], 'Loyal Customers': [325, 334, 335, 343, 344, 345, 352, 353, 354, 355, 425, 434, 435, 443, 444, 445, 452, 453, 454, 455], 'Potential Loyalist': [413, 414, 415, 422, 423, 424, 431, 432, 433, 441, 442, 451, 513, 514, 522, 523, 531, 532, 541], 'New Customers': [511, 512, 521], 'Promising': [411, 412, 421], 'Need Attention': [311, 312, 313, 321, 322, 331], 'About To Sleep': [314, 315, 323, 324, 332, 333, 341, 342, 351], 'At Risk': [115, 124, 125, 133, 134, 135, 142, 143, 144, 151, 152, 153, 215, 224, 225, 233, 234, 235, 242, 243, 244, 251, 252, 253], "Can't Lose Them": [145, 154, 155, 245, 254, 255], 'Hibernating': [111, 112, 113, 114, 121, 122, 123, 131, 132, 141, 211, 212, 213, 214, 221, 222, 223, 231, 232, 241]}
def apply_classify(val):
    if val in score_dict['Champions']: return 'Champions'
    if val in score_dict['Loyal Customers']: return 'Loyal Customers'
    if val in score_dict['Potential Loyalist']: return 'Potential Loyalist'
    if val in score_dict['New Customers']: return 'New Customers'
    if val in score_dict['Need Attention']: return 'Need Attention'
    if val in score_dict['About To Sleep']: return 'About To Sleep'
    if val in score_dict['At Risk']: return 'At Risk'
    if val in score_dict['Can\'t Lose Them']: return 'Can\'t Lose Them'
    if val in score_dict['Hibernating']: return 'Hibernating'
    if val in score_dict['Promising']: return 'Promising'
    return ""
def rfm_analysis(df: pd.DataFrame, timestamp: str, monetary: str, customer: str) -> pd.DataFrame:
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
    rfm_df = rfm_df.round(0)
    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)

    # normalizing the rank of the customers
    rfm_df['R_Score'] = np.ceil((rfm_df['R_rank']/rfm_df['R_rank'].max())*5).astype(int)
    rfm_df['F_Score'] = np.ceil((rfm_df['F_rank']/rfm_df['F_rank'].max())*5).astype(int)
    rfm_df['M_Score'] = np.ceil((rfm_df['M_rank']/rfm_df['M_rank'].max())*5).astype(int)
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    rfm_df['RFM_Score'] = 100*rfm_df['R_Score'] + 10 *rfm_df['F_Score']+ rfm_df['M_Score']

    rfm_df["Customer_segment"] = rfm_df['RFM_Score'].apply(apply_classify)
    rfm_df = rfm_df.sort_values(['R_Score', 'F_Score', 'M_Score'], ascending = [False, False, False])

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
    

def get_mapping(user_columns: list[str]):
    """
    Return mapping of user columns to system columns and store as JSON file.

    Parameters:
        user_columns (list(str)): User columns
    
    Returns:
        map (dict): Mapping of user columns to system columns
    """
    # default_mapping = {column: column for column in user_columns}

    # prompt = f"""Map the following columns with the list that has the same meaning, if not found, keep the original name of the input column:

    #     Columns: {user_columns}

    #     JSON list: {settings.TARGET_LIST}

    #     Just respond in JSON-like format."""
    
    # for attempt in range(settings.LIMIT_OPENAPI_CALLS):  # Retry up to 5 times
    #     try:
    #         response = openai_client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": prompt},
    #             ]
    #         )
    #         try:
    #             message = response.choices[0].message.content
    #             start = message.find('{')
    #             end = message.find('}') + 1
    #             res = json.loads(message[start: end])
    #             input_token = response.usage.prompt_tokens
    #             output_token = response.usage.completion_tokens
    #             price = input_token * INPUT_PRICE + output_token * OUTPUT_PRICE
    #             print("Cost:", price, "$")
    #             return res
    #         except json.JSONDecodeError:
    #             print("Failed to decode JSON response. Returning default mapping.")
    #             return default_mapping
    #     except RateLimitError:
    #         print(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
    #         time.sleep(2 ** attempt)  # Exponential backoff
    #         continue
    #     except Exception as e:
    #         print(f"Unexpected error: {str(e)}. Returning default mapping.")
    #         return default_mapping
    
    # print("Max retries reached. Returning default mapping.")
    # return default_mapping

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
    # return {
    #         'Invoice':'Transaction ID',
    #         'StockCode': 'Product ID',
    #         'Description': 'Description',
    #         'Quantity': 'Quantity',
    #         'InvoiceDate': 'Date',
    #         'Price': 'Unit Price',
    #         'Customer ID': 'Customer ID',
    #         'Country': 'Store Location',
    #         'Total Price': 'Total Price',
    # }

def train_with_prophet(data, test_size, target):
    # Prepare the data for Prophet
    model = ProphetModel()
    model.train(test_size, data, target)
    return model

def train_with_xgboost(data, test_size, target, time_range, lag_size=30):
    # Prepare the data for Prophet
    model = XGBoostModel(lag_size, time_range)
    model.train(test_size, data, target)
    return model

def train_with_lstm(data, test_size, target, time_range, lag_size=30):
    model = LSTMModel(lag_size, time_range)
    model.train(test_size, data, target)
    return model

def filter_data(data, filters):
    for key, value in filters.items():
        data = data[data[key] == value]
    return data

def create_sql_engine(user_id):
    directory = 'database'
    file_name = f'{user_id}.db'
    file_path = os.path.join(directory, file_name)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Create the SQLAlchemy engine with the path to the SQLite database file
    engine = create_engine(f"sqlite:///{file_path}")
    drop_table_query = "DROP TABLE IF EXISTS data"

    # Execute the drop table query using the engine
    with engine.connect() as connection:
        connection.execute(text(drop_table_query))
    return engine
