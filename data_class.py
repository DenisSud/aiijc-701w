import sqlite3
import os
import re
import pandas as pd
from langdetect import detect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

class Data:
    def __init__(self, db_path: str = "data/data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS train (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                answer TEXT,
                language TEXT
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT
            )
        """)
        
        self.conn.commit()
    
    def load_csv_to_db(self):
        if os.path.exists("data/train.csv"):
            train_df = pd.read_csv("data/train.csv")
            train_df.reset_index(drop=True, inplace=True)
            train_df.index.name = 'id'
            train_df.reset_index(inplace=True)
            train_df.to_sql('train', self.conn, if_exists='replace', index=False)
        
        if os.path.exists("data/test_public.csv"):
            test_df = pd.read_csv("data/test_public.csv")
            test_df.reset_index(drop=True, inplace=True)
            test_df.index.name = 'id'
            test_df.reset_index(inplace=True)
            test_df.to_sql('test', self.conn, if_exists='replace', index=False)
    
    def add_language_column(self):
        try:
            self.cursor.execute("ALTER TABLE train ADD COLUMN language TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def detect_languages(self):
        self.cursor.execute("SELECT id, task FROM train")
        rows = self.cursor.fetchall()
        
        for row_id, task in rows:
            if task and isinstance(task, str):
                cleaned_task = re.sub(r'[\d+\-*/%=()<>!&|^~]×', '', task).strip()
                
                if not cleaned_task:  
                    lang = 'num'
                else:
                    try:
                        lang = detect(cleaned_task) if cleaned_task else 'num'
                    except LangDetectException:
                        lang = 'num'
                
                self.cursor.execute(
                    "UPDATE train SET language = ? WHERE id = ?",
                    (lang, row_id)
                )
        
        self.conn.commit()

    def add_column(self, table_name: str, column_name: str, column_type: str = "TEXT"):
        """Добавляет новый столбец в указанную таблицу"""
        try:
            self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def get_data_for_translation_filtered(self, table_name: str = 'train', 
                                         source_column: str = 'task',
                                         target_column: str = None) -> list:
        if target_column:
            query = f"""SELECT id, {source_column} FROM {table_name} 
                       WHERE {source_column} IS NOT NULL 
                       AND ({target_column} IS NULL OR {target_column} = '')"""
        else:
            query = f"SELECT id, {source_column} FROM {table_name} WHERE {source_column} IS NOT NULL"
        
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def prepare_translation_column(self, table_name: str = 'train', 
                                 target_column: str = 'translated_google'):
        """Подготавливает столбец для записи переводов"""
        self.add_column(table_name, target_column)
    
    def save_translated_text(self, table_name: str, target_column: str, 
                           row_id: int, translated_text: str):
        """Сохраняет переведенный текст в базу данных"""
        update_query = f"UPDATE {table_name} SET {target_column} = ? WHERE id = ?"
        self.cursor.execute(update_query, (translated_text, row_id))
    
    def commit_changes(self):
        self.conn.commit()
        
    def get_train_data(self):
        return pd.read_sql_query("SELECT * FROM train", self.conn)
    
    def get_test_data(self):
        return pd.read_sql_query("SELECT * FROM test", self.conn)
    
    def close(self):
        self.conn.close()


def main():
    data_manager = Data()
    data_manager.create_tables()
    data_manager.load_csv_to_db()
    
    print("Определение языков...")
    data_manager.add_language_column()
    data_manager.detect_languages()

#main()