import pymongo


class Database(object):
    URI = "mongodb+srv://admin:admin@cluster0.50bid.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    DATABASE = None

    @staticmethod
    def initialize():  # Initializes Database (Mongodb must be already running on system)
        client = pymongo.MongoClient(Database.URI)
        Database.DATABASE = client["fullstack"]

    @staticmethod
    def insert(
        collection, data
    ):  # Inserts new record in db.collection (data must be in JSON)
        Database.DATABASE[collection].insert(data)

    @staticmethod
    def insert_many(
        collection, data
    ):  # Inserts new record in db.collection (data must be in JSON)
        Database.DATABASE[collection].insert_many(data)

    @staticmethod
    def find(
        collection, query
    ):  # Returns all records from db.collection matching query
        return Database.DATABASE[collection].find(query)  # query must be in JSON

    @staticmethod
    def getCollectionList():
        return Database.DATABASE.list_collection_names()

    @staticmethod
    def find_one(
        collection, query
    ):  # Returns fist record from db.collection matching query
        return Database.DATABASE[collection].find_one(query)  # query must be in JSON

    @staticmethod
    def update(
        collection, query, data
    ):  # Modifies record matching query in db.collection
        # (upsert = true): creates a new record when no record matches the query criteria
        Database.DATABASE[collection].update(query, data, upsert=True)

    @staticmethod
    def remove(collection, query):  # Deletes record from db.collecion
        Database.DATABASE[collection].remove(query)

    @staticmethod
    def findmax(collection, field):  # Deletes record from db.collecion
        return Database.DATABASE[collection].find().sort(field, -1).limit(1)

    @staticmethod
    def drop(collection):  # Deletes record from db.collecion
        return Database.DATABASE[collection].drop()
