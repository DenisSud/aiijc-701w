import sqlite3
import os
import re
import pandas as pd
from langdetect import detect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import sqlitecloud

DetectorFactory.seed = 0

class Data:
    def __init__(self, db_path: str = None, cloud_path: str = None):
        if db_path:
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
        if cloud_path:
            self.db_path = cloud_path
            self.conn = sqlitecloud.connect(cloud_path)
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
            train_df.index.name = "id"
            train_df.reset_index(inplace=True)

            if self.db_path.startswith("sqlitecloud://"):
                self._insert_df_to_cloud(train_df, "train")
            else:
                train_df.to_sql("train", self.conn, if_exists="replace", index=False)

        if os.path.exists("data/test_public.csv"):
            test_df = pd.read_csv("data/test_public.csv")
            test_df.reset_index(drop=True, inplace=True)
            test_df.index.name = "id"
            test_df.reset_index(inplace=True)

            if self.db_path.startswith("sqlitecloud://"):
                self._insert_df_to_cloud(test_df, "test")
            else:
                test_df.to_sql("test", self.conn, if_exists="replace", index=False)

    def _insert_df_to_cloud(self, df, table_name):
        """Вспомогательный метод для вставки данных в SQLiteCloud"""
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            columns = []
            for col, dtype in zip(df.columns, df.dtypes):
                sql_type = "TEXT" if dtype == "object" else "INTEGER"
                columns.append(f"{col} {sql_type}")

            create_sql = f"""
                CREATE TABLE {table_name} (
                    {", ".join(columns)}
                )
            """
            self.cursor.execute(create_sql)

            placeholders = ", ".join(["?"] * len(df.columns))
            columns_str = ", ".join(df.columns)
            insert_sql = (
                f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            )

            batch_size = 100
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                self.cursor.executemany(insert_sql, batch.values.tolist())

            self.conn.commit()
            print(f"Данные успешно загружены в таблицу {table_name} в SQLiteCloud")

        except Exception as e:
            print(f"Ошибка при загрузке данных в SQLiteCloud: {e}")
            self.conn.rollback()

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
                cleaned_task = re.sub(r"[\d+\-*/%=()<>!&|^~]×", "", task).strip()

                if not cleaned_task:
                    lang = "num"
                else:
                    try:
                        lang = detect(cleaned_task) if cleaned_task else "num"
                    except LangDetectException:
                        lang = "num"

                self.cursor.execute(
                    "UPDATE train SET language = ? WHERE id = ?", (lang, row_id)
                )

        self.conn.commit()

    def add_column(self, table_name: str, column_name: str, column_type: str = "TEXT"):
        """Добавляет новый столбец в указанную таблицу"""
        try:
            self.cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def get_data_for_translation(self, table_name: str = "train", 
                                             source_column: str = "task",
                                             target_column: str = None) -> list:
        """Получает данные для перевода вместе с языком оригинального текста"""
        if target_column:
            query = f"""SELECT id, {source_column}, language FROM {table_name} 
                       WHERE {source_column} IS NOT NULL 
                       AND ({target_column} IS NULL OR {target_column} = '')"""
        else:
            query = f"SELECT id, {source_column}, language FROM {table_name} WHERE {source_column} IS NOT NULL"
        
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def prepare_translation_column(
        self, table_name: str = "train", target_column: str = "translated_google"
    ):
        """Подготавливает столбец для записи переводов"""
        self.add_column(table_name, target_column)

    def save_translated_text(
        self, table_name: str, target_column: str, row_id: int, translated_text: str
    ):
        """Сохраняет переведенный текст в базу данных"""
        update_query = f"UPDATE {table_name} SET {target_column} = ? WHERE id = ?"
        self.cursor.execute(update_query, (translated_text, row_id))

    def commit_changes(self):
        self.conn.commit()

    def get_train_data(self):
        if self.db_path.startswith("sqlitecloud://"):
            self.cursor.execute("SELECT * FROM train")
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(rows, columns=columns)

        return pd.read_sql_query("SELECT * FROM train", self.conn)

    def get_test_data(self):
        if self.db_path.startswith("sqlitecloud://"):
            self.cursor.execute("SELECT * FROM test")
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(rows, columns=columns)
        return pd.read_sql_query("SELECT * FROM test", self.conn)

    def get_eval_data(self):
        if self.db_path.startswith("sqlitecloud://"):
            self.cursor.execute("SELECT * FROM evaluations")
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return pd.DataFrame(rows, columns=columns)
        return pd.read_sql_query("SELECT * FROM evaluations", self.conn)

    def close(self):
        self.conn.close()

    def merge_databases(
        self, other_db_path: str, table_name: str, merged_db_path: str = None
    ):
        """
        Объединяет текущую базу данных с другой базой данных, обрабатывая уникальные индексы

        :param other_db_path: путь ко второй базе данных
        :param table_name: имя таблицы для объединения
        :param merged_db_path: путь для сохранения объединенной БД (None = использовать текущую)
        """
        try:
            self.conn.execute(f"ATTACH DATABASE '{other_db_path}' AS other_db")

            self.cursor.execute(f"PRAGMA table_info({table_name})")
            main_columns = {col[1] for col in self.cursor.fetchall()}

            self.cursor.execute(f"PRAGMA other_db.table_info({table_name})")
            other_columns = {col[1] for col in self.cursor.fetchall()}

            if main_columns != other_columns:
                print("Ошибка: структура таблиц не совпадает")
                print(f"Основная БД: {main_columns}")
                print(f"Другая БД: {other_columns}")
                return False

            self.cursor.execute(f"SELECT MAX(eval_id) FROM {table_name}")
            max_id = self.cursor.fetchone()[0] or 0

            if merged_db_path:
                new_conn = sqlite3.connect(merged_db_path)
                new_cursor = new_conn.cursor()

                self.cursor.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                )
                create_table_sql = self.cursor.fetchone()[0]
                new_cursor.execute(create_table_sql)

                new_cursor.execute(
                    f"INSERT INTO {table_name} SELECT * FROM main.{table_name}"
                )

                new_cursor.execute(f"""
                    INSERT INTO {table_name} 
                    SELECT 
                        eval_id + {max_id} AS new_id, 
                        other_columns...
                    FROM other_db.{table_name}
                """)
                new_conn.commit()
                new_conn.close()

            else:
                self.cursor.execute(
                    f"CREATE TEMP TABLE temp_merge AS SELECT * FROM other_db.{table_name}"
                )
                self.cursor.execute(
                    f"UPDATE temp_merge SET eval_id = eval_id + {max_id}"
                )
                self.cursor.execute(
                    f"INSERT INTO {table_name} SELECT * FROM temp_merge"
                )
                self.cursor.execute("DROP TABLE temp_merge")
                self.conn.commit()

            self.conn.execute("DETACH DATABASE other_db")
            return True

        except sqlite3.Error as e:
            print(f"Ошибка при объединении баз данных: {e}")
            return False

    def execute(self, query: str):
        return self.cursor.execute(query, self.conn)
    

def main():
    data_manager = Data()
    data_manager.create_tables()
    data_manager.load_csv_to_db()

    print("Определение языков...")
    data_manager.add_language_column()
    data_manager.detect_languages()

#main()