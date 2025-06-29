
# Conversational Database Agent with Natural Language Query Mapping to MongoDB

# Step 1: Install Required Packages

# Step 2: Load Environment Variables
from typing import List, Dict, Any
from crewai import Task, Crew, tools
from pymongo.collection import Collection
from langchain_core.messages import HumanMessage
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOEKN")

# Step 3: Setup MongoDB Connection
from pymongo import MongoClient
client = MongoClient(MONGODB_URI)
db = client.sample_analytics
conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Step 4: Define Query Execution Interface
from crewai.tools import tool


def get_definition(entity):
    definitions = definitions = {
    "account": "An account represents a customer's financial relationship and includes limits and product types.",
    "transaction": "A transaction is a financial activity like deposits or withdrawals by a customer.",
    
    # Accounts collection
    "limit": "The maximum available credit or transaction limit on the account. It represents the financial ceiling for transactions such as withdrawals or trades.",
    "products": "A list of financial services associated with the account, such as 'brokerage', 'investment', or 'commodity'.",

    # Customers collection
    "tier_and_details": "An embedded document providing details about the customer's membership tier, associated benefits, and whether the tier is currently active.",
    "tier": "The membership level of the customer, such as 'Silver', 'Gold', or 'Platinum', indicating their service level.",
    "benefits": "A list of perks associated with the customer's tier, such as 'priority_support' or 'lower_fees'.",
    "active": "A boolean flag that indicates whether the customer's tier benefits are currently active or inactive.",

    # Transactions collection
    "transaction_count": "The total number of individual transactions stored within a single transaction bucket document.",
    "bucket_start_date": "The earliest date of the transactions stored in a bucket, used for time-based partitioning.",
    "bucket_end_date": "The latest date of the transactions stored in a bucket, used for time-based partitioning.",
    "transaction_code": "A short identifier for the type of transaction. Valid values include 'buy' and 'sell'.",
    "symbol": "The asset's ticker symbol involved in the transaction. Common examples include 'sap', 'team', 'nflx', 'ibm', 'adbe', and 'msft'."
}

    return definitions.get(entity.lower(), "No definition found.")


# All MongoDB functions follow the projection pattern:
# 1 to include a field
# 0 to exclude a field like _id

#functions for accounts section :-
from typing import List, Dict, Any
from datetime import datetime
from pymongo.collection import Collection # You'll need to ensure 'db' is available
                                        # when these classes are instantiated.
from crewai.tools import BaseTool
def get_definition(entity: str) -> str:
    """Provides definitions for financial terms."""
    definitions = {
        "account": "An account represents a customer's financial relationship and includes limits and product types.",
        "transaction": "A transaction is a financial activity like deposits or withdrawals by a customer.",
        "limit": "The maximum available credit or transaction limit on the account. It represents the financial ceiling for transactions such as withdrawals or trades.",
        "products": "A list of financial services associated with the account, such as 'brokerage', 'investment', or 'commodity'.",
        "tier_and_details": "An embedded document providing details about the customer's membership tier, associated benefits, and whether the tier is currently active.",
        "tier": "The membership level of the customer, such as 'Silver', 'Gold', or 'Platinum', indicating their service level.",
        "benefits": "A list of perks associated with the customer's tier, such as 'priority_support' or 'lower_fees'.",
        "active": "A boolean flag that indicates whether the customer's tier benefits are currently active or inactive.",
        "transaction_count": "The total number of individual transactions stored within a single transaction bucket document.",
        "bucket_start_date": "The earliest date of the transactions stored in a bucket, used for time-based partitioning.",
        "bucket_end_date": "The latest date of the transactions stored in a bucket, used for time-based partitioning.",
        "transaction_code": "A short identifier for the type of transaction. Valid values include 'buy' and 'sell'.",
        "symbol": "The asset's ticker symbol involved in the transaction. Common examples include 'sap', 'team', 'nflx', 'ibm', 'adbe', and 'msft'."
    }
    return definitions.get(entity.lower(), "No definition found.")

# --- Custom BaseTool Classes ---

# Definition Tool
class DefinitionLookupTool(BaseTool):
    name: str = "DefinitionLookup"
    description: str = "Returns definition of financial terms like account, transaction, limit, products, tier, benefits, active, transaction_count, bucket_start_date, bucket_end_date, transaction_code, or symbol."

    def _run(self, entity: str) -> str:
        """
        Retrieves the definition of a given financial entity.
        Args:
            entity (str): The financial term to look up (e.g., 'account', 'limit').
        Returns:
            str: The definition of the entity, or 'No definition found.' if not available.
        """
        return get_definition(entity)

# Accounts Tools
class GetAccountLimitTool(BaseTool):
    name: str = "GetAccountLimit"
    description: str = "Retrieve the credit or transaction limit for a given account ID."

    def _run(self, account_id: int) -> str:
        """
        Retrieves the credit or transaction limit of a given account.
        Args:
            account_id (int): The ID of the account.
        Returns:
            str: The limit of the account as a string, or an error message if not found.
        """
        # Ensure db is accessible, e.g., through import or constructor
        global db # Assuming db is a global from main.py for simplicity in this conversion
        doc = db.accounts.find_one({"account_id": account_id})
        return str(doc["limit"]) if doc and "limit" in doc else "Limit not found."

class GetAccountProductsTool(BaseTool):
    name: str = "GetAccountProducts"
    description: str = "List all financial products associated with a specific account ID."

    def _run(self, account_id: int) -> str:
        """
        Returns the list of financial products associated with a specific account.
        Args:
            account_id (int): The ID of the account.
        Returns:
            str: A string representation of the list of products, or an empty list if none found.
        """
        global db
        doc = db["accounts"].find_one({"account_id": account_id}, {"products": 1})
        if doc and "products" in doc:
            return str(doc["products"])
        return "[]"

class GetAccountsByProductTool(BaseTool):
    name: str = "GetAccountsByProduct"
    description: str = "Find all accounts that include a specific product type (e.g., 'Derivatives', 'brokerage', 'investment', 'commodity')."

    def _run(self, product_name: str) -> str:
        """
        Return all accounts that include a specific product type.
        Args:
            product_name (str): The name of the product (e.g., 'brokerage').
        Returns:
            str: A string representation of a list of accounts.
        """
        global db
        return str(list(db.accounts.find({"products": product_name}, {"_id": 0})))

class GetHighLimitAccountsTool(BaseTool):
    name: str = "GetHighLimitAccounts"
    description: str = "List accounts that have a credit or transaction limit above the given threshold. Default threshold is 100000."

    def _run(self, threshold: int = 100000) -> str:
        """
        Return all accounts with a limit greater than the given threshold.
        Args:
            threshold (int, optional): The minimum limit for accounts to be returned. Defaults to 100000.
        Returns:
            str: A string representation of a list of high-limit accounts.
        """
        global db
        return str(list(db.accounts.find({"limit": {"$gt": threshold}}, {"_id": 0})))

class ListAllAccountIDsTool(BaseTool):
    name: str = "ListAllAccountIDs"
    description: str = "Get a list of all account IDs in the system."

    def _run(self) -> str:
        """
        Return a list of all account IDs in the system.
        Returns:
            str: A string representation of a list of all account IDs.
        """
        global db
        return str([doc["account_id"] for doc in db.accounts.find({}, {"account_id": 1})])

# Customer Tools
class GetCustomerByUsernameTool(BaseTool):
    name: str = "GetCustomerByUsername"
    description: str = "Get customer details by username."

    def _run(self, username: str) -> str:
        """
        Retrieves details for a customer given their username.
        Args:
            username (str): The username of the customer.
        Returns:
            str: A string representation of the customer document or None if not found.
        """
        global db
        return str(db.customers.find_one({"username": username}, {"_id": 0}))

class GetCustomersWithEmailDomainTool(BaseTool):
    name: str = "GetCustomersWithEmailDomain"
    description: str = "List customers with a specific email domain (e.g., 'example.com')."

    def _run(self, domain: str) -> str:
        """
        Lists customers whose email addresses belong to a specific domain.
        Args:
            domain (str): The email domain (e.g., 'gmail.com').
        Returns:
            str: A string representation of a list of customers with matching email domains.
        """
        global db
        return str(list(db.customers.find({"email": {"$regex": f"@{domain}$"}}, {"name": 1, "email": 1, "_id": 0})))

class GetCustomersByAccountTool(BaseTool):
    name: str = "GetCustomersByAccount"
    description: str = "Find customers owning a specific account ID."

    def _run(self, account_id: int) -> str:
        """
        Finds customers associated with a given account ID.
        Args:
            account_id (int): The ID of the account.
        Returns:
            str: A string representation of a list of customers associated with the account.
        """
        global db
        return str(list(db.customers.find({"accounts": account_id}, {"username": 1, "name": 1, "accounts": 1, "_id": 0})))

class GetAccountsByCustomerTool(BaseTool):
    name: str = "GetAccountsByCustomer"
    description: str = "Find accounts of customer by their specific username."

    def _run(self, username: str) -> str:
        """
        Finds account details for a customer given their username.
        Args:
            username (str): The username of the customer.
        Returns:
            str: A string representation of the customer's account information.
        """
        global db
        return str(list(db.customers.find({"username": username}, {"username": 1, "name": 1, "accounts": 1, "_id": 0})))

class GetCustomerTiersTool(BaseTool):
    name: str = "GetCustomerTiers"
    description: str = "Lists tier and benefit information of all customers."

    def _run(self) -> str:
        """
        Retrieves tier and benefit information for all customers.
        Returns:
            str: A string representation of a list of customer tier details.
        """
        global db
        return str(list(db.customers.find({}, {"username": 1, "tier_and_details": 1, "_id": 0})))

class GetCustomersByBirthYearTool(BaseTool):
    name: str = "GetCustomersByBirthYear"
    description: str = "Get customers born in a specific year."

    def _run(self, year: int) -> str:
        """
        Retrieves customers born in a specific year.
        Args:
            year (int): The birth year to query.
        Returns:
            str: A string representation of a list of customers born in the specified year.
        """
        global db
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        return str(list(db.customers.find({"birthdate": {"$gte": start_date, "$lt": end_date}}, {"name": 1, "birthdate": 1, "_id": 0})))

class GetAccountsByNameOrUsernameTool(BaseTool):
    name: str = "GetAccountsByNameOrUsername"
    description: str = "Returns a list of account IDs for a customer using their name or username."

    def _run(self, name_or_username: str) -> str:
        """
        Returns a list of account IDs for a customer identified by name or username.
        Args:
            name_or_username (str): The name or username of the customer.
        Returns:
            str: A string representation of a list of account IDs.
        """
        global db
        customer = db["customers"].find_one({
            "$or": [
                {"name": name_or_username},
                {"username": name_or_username}
            ]
        }, {"customer_id": 1})

        if not customer:
            return "[]"

        customer_id = customer["customer_id"]
        accounts = db["accounts"].find(
            {"customer_id": customer_id},
            {"account_id": 1}
        )
        return str([acc["account_id"] for acc in accounts])

# Transaction Tools
class GetTransactionsByAccountTool(BaseTool):
    name: str = "GetTransactionsByAccount"
    description: str = "Retrieve all transactions for a specific account, optionally filtered by start_date and/or end_date in YYYY-MM-DD format."

    def _run(self, account_id: int, start_date: str = None, end_date: str = None) -> str:
        """
        Retrieves transactions for a specific account, with optional date filtering.
        Args:
            account_id (int): The ID of the account.
            start_date (str, optional): The start date for filtering transactions (YYYY-MM-DD).
            end_date (str, optional): The end date for filtering transactions (YYYY-MM-DD).
        Returns:
            str: A string representation of the list of transactions.
        """
        global db
        match = {"account_id": account_id}
        if start_date or end_date:
            match["transactions"] = {
                "$elemMatch": {
                    "date": {
                        **({"$gte": datetime.strptime(start_date, "%Y-%m-%d")} if start_date else {}),
                        **({"$lte": datetime.strptime(end_date, "%Y-%m-%d")} if end_date else {})
                    }
                }
            }
        buckets = list(db.transactions.find(match))
        transactions = []
        for bucket in buckets:
            for tx in bucket["transactions"]:
                tx_date = tx["date"]
                if (
                    (not start_date or tx_date >= datetime.strptime(start_date, "%Y-%m-%d")) and
                    (not end_date or tx_date <= datetime.strptime(end_date, "%Y-%m-%d"))
                ):
                    transactions.append(tx)
        return str(transactions)

class GetTransactionSummaryByTypeTool(BaseTool):
    name: str = "GetTransactionSummaryByType"
    description: str = "Summarize the total number and value of each transaction type (e.g., 'buy', 'sell') for a given account."

    def _run(self, account_id: int) -> str:
        """
        Summarizes transactions by type (count and total value) for a given account.
        Args:
            account_id (int): The ID of the account.
        Returns:
            str: A string representation of the transaction summary.
        """
        global db
        buckets = list(db.transactions.find({"account_id": account_id}))
        summary = {}
        for bucket in buckets:
            for tx in bucket["transactions"]:
                code = tx["transaction_code"]
                if code not in summary:
                    summary[code] = {"count": 0, "total_value": 0}
                summary[code]["count"] += 1
                summary[code]["total_value"] += float(tx["total"])
        return str(summary)

class GetTransactionsBySymbolTool(BaseTool):
    name: str = "GetTransactionsBySymbol"
    description: str = "Get all transactions involving a specific stock or currency symbol (e.g., 'sap', 'team', 'nflx', 'ibm', 'adbe', 'msft') for a given account."

    def _run(self, account_id: int, symbol: str) -> str:
        """
        Retrieves all transactions for a specific account involving a particular symbol.
        Args:
            account_id (int): The ID of the account.
            symbol (str): The stock or currency symbol.
        Returns:
            str: A string representation of the list of transactions matching the symbol.
        """
        global db
        buckets = list(db.transactions.find({"account_id": account_id}))
        return str([
            tx for bucket in buckets
            for tx in bucket["transactions"]
            if tx["symbol"].lower() == symbol.lower()
        ])

class GetTransactionVolumeOverTimeTool(BaseTool):
    name: str = "GetTransactionVolumeOverTime"
    description: str = "Aggregate transaction count and total value over time (grouped by 'month' or 'year') for a given account."

    def _run(self, account_id: int, group_by: str = "month") -> str:
        """
        Aggregates transaction volume (count and total value) over time for an account.
        Args:
            account_id (int): The ID of the account.
            group_by (str, optional): How to group the data ('month' or 'year'). Defaults to 'month'.
        Returns:
            str: A string representation of the aggregated transaction volume.
        """
        global db
        assert group_by in ["month", "year"], "group_by must be 'month' or 'year'"
        buckets = list(db.transactions.find({"account_id": account_id}))
        result = {}
        for bucket in buckets:
            for tx in bucket["transactions"]:
                dt = tx["date"]
                key = f"{dt.year}" if group_by == "year" else f"{dt.year}-{dt.month:02d}"
                if key not in result:
                    result[key] = {"count": 0, "total_value": 0}
                result[key]["count"] += 1
                result[key]["total_value"] += float(tx["total"])
        return str(result)

class FallbackHandlerTool(BaseTool):
    name: str = "FallbackHandler"
    description: str = "Used when no other tools match or the input seems irrelevant. Provides a default helpful message."

    def _run(self, query: str = "") -> str:
        """
        Handles queries that cannot be addressed by other tools.
        Args:
            query (str): The original query that couldn't be handled. (Optional, for context)
        Returns:
            str: A default fallback message.
        """
        return "I'm sorry, I couldn't find an answer to your query. Try asking other questions related to accounts, transactions, or customers."

# --- Instantiate all the tools ---

# Definition Tool
DefinitionLookupTool = DefinitionLookupTool()

# Accounts Tools
GetAccountLimitTool = GetAccountLimitTool()
GetAccountProductsTool = GetAccountProductsTool()
GetAccountsByProductTool = GetAccountsByProductTool()
GetHighLimitAccountsTool = GetHighLimitAccountsTool()
ListAllAccountIDsTool = ListAllAccountIDsTool()

# Customer Tools
GetCustomerByUsernameTool = GetCustomerByUsernameTool()
GetCustomersWithEmailDomainTool = GetCustomersWithEmailDomainTool()
GetCustomersByAccountTool = GetCustomersByAccountTool()
GetAccountsByCustomerTool = GetAccountsByCustomerTool()
GetCustomerTiersTool = GetCustomerTiersTool()
GetCustomersByBirthYearTool = GetCustomersByBirthYearTool()
GetAccountsByNameOrUsernameTool = GetAccountsByNameOrUsernameTool()

# Transaction Tools
GetTransactionsByAccountTool = GetTransactionsByAccountTool()
GetTransactionSummaryByTypeTool = GetTransactionSummaryByTypeTool()
GetTransactionsBySymbolTool = GetTransactionsBySymbolTool()
GetTransactionVolumeOverTimeTool = GetTransactionVolumeOverTimeTool()

# Fallback Tool
FallbackHandlerTool = FallbackHandlerTool()

# Note: If you also have the IntentMatcherTool class, you would instantiate it like this:

# Step 6: Set up Conversational Agent
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings



# Step 3: Setup MongoDB Connection
from pymongo import MongoClient
client = MongoClient(MONGODB_URI)

db = client.sample_analytics
from crewai import LLM
llm = LLM(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

from crewai import Agent

from difflib import get_close_matches
import json

with open("sample_questions.json", "r") as f:
    sample_questions_data = json.load(f)

def match_intent(query: str) -> str:
    """
    Matches a user query to known intents using similarity with sample questions.
    Args:
        query (str): The user's natural language query.
    Returns:
        str: A string indicating the detected intent and matched question, or "No close intent match found.".
    """
    questions = [q["question"] for q in sample_questions_data]
    match = get_close_matches(query, questions, n=1, cutoff=0.6) # from difflib
    if match:
        for item in sample_questions_data:
            if item["question"] == match[0]:
                return f"Detected intent: {item['intent']} (matched question: {item['question']})"
    return "No close intent match found."

# --- Custom BaseTool Class for Intent Matching ---

class IntentMatcherTool(BaseTool):
    name: str = "IntentMatcher"
    description: str = "Matches user query to known intents using similarity with sample questions. Returns the detected intent and the matched sample question."
    # You might want to pass sample_questions_data as a field if the tool is instantiated in a different scope
    # sample_questions_data: List[Dict[str, Any]] # Example of adding data as a Pydantic field

    def _run(self, query: str) -> str:
        """
        Executes the intent matching logic.
        Args:
            query (str): The user's natural language query.
        Returns:
            str: The result of the intent matching.
        """
        return match_intent(query)
IntentMatcherTool = IntentMatcherTool()
# CrewAI Agents using shared llm
class ComplexQueryExecutorTool(BaseTool):
    name: str = "ComplexQueryExecutor"
    description: str = "Executes a sequence of MongoDB operations based on a plan, using results from previous steps."

    def _run(self, plan: List[Dict[str, Any]]) -> str:
        """
        Executes a sequence of operations.
        'plan' is a list of dictionaries, each describing an operation.
        Example: [{'tool': 'GetAccountsByCustomer', 'params': {'username': 'JohnDoe'}},
                  {'tool': 'GetTransactionSummaryByType', 'params': {'account_id': 'PREVIOUS_RESULT_ITEM.account_id'}}]
        """
        global db # Ensure db is accessible

        all_results = []
        for step in plan:
            tool_name = step.get("tool")
            params = step.get("params", {})
            current_result = None

            # Resolve parameters that depend on previous results
            resolved_params = {}
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("PREVIOUS_RESULT_ITEM"):
                    # This is a placeholder for actual logic to parse and retrieve
                    # data from previous_step_results.
                    # You'll need more sophisticated parsing here depending on your
                    # 'PREVIOUS_RESULT_ITEM' format.
                    # For simplicity, let's assume 'PREVIOUS_RESULT_ITEM.account_id' means
                    # extract 'account_id' from the last result if it was a list of dicts.
                    resolved_params[key] = self._resolve_previous_result(value, all_results)
                else:
                    resolved_params[key] = value

            # Dynamically call the tool
            # This requires a mapping from tool_name string to actual tool instance
            tool_instance = globals().get(tool_name) # Accesses global tool instances
            if tool_instance and hasattr(tool_instance, '_run'):
                try:
                    # You need to carefully handle how tool_instance._run expects arguments
                    # It might require unpacking resolved_params as **resolved_params
                    current_result = tool_instance._run(**resolved_params)
                    all_results.append({tool_name: current_result})
                except TypeError as e:
                    return f"Error executing tool {tool_name} with params {resolved_params}: {e}"
            else:
                return f"Tool {tool_name} not found or not callable."
        return str(all_results)

    def _resolve_previous_result(self, param_str: str, all_results: List[Dict[str, Any]]) -> Any:
        """Helper to extract data from previous results."""
        if not all_results:
            return None # Or raise an error, depending on expected behavior

        # Simplified example: assuming the last result is a list of dicts
        # and we are looking for 'account_id' from those dicts.
        # This logic needs to be robust for various previous result formats.
        if "PREVIOUS_RESULT_ITEM.account_id" in param_str:
            last_result_dict = all_results[-1] # Get the last result
            # Find the first value that looks like a list of dictionaries
            for key, value in last_result_dict.items():
                if isinstance(value, str):
                    try:
                        # Attempt to parse string representation of list of dicts
                        parsed_value = eval(value) # Using eval is risky, consider json.loads
                        if isinstance(parsed_value, list) and all(isinstance(item, dict) for item in parsed_value):
                            return [item.get('account_id') for item in parsed_value if 'account_id' in item]
                    except Exception:
                        pass # Not a parsable list of dicts
            return None
        return None # Fallback if not recognized

# Instantiate the new tool
ComplexQueryExecutorTool = ComplexQueryExecutorTool()

class RecallChatHistoryTool(BaseTool):
    name: str = "RecallChatHistory"
    description: str = "Recalls the previous questions asked by the user in the current conversation."
    conversational_memory: ConversationBufferMemory # Pass memory to the tool

    def _run(self) -> str:
        history = self.conversational_memory.load_memory_variables({})["chat_history"]
        user_questions = []
        for i, message in enumerate(history):
            if isinstance(message, HumanMessage): # Assuming HumanMessage from langchain.schema
                user_questions.append(f"Question {len(user_questions) + 1}: {message.content}")
        if user_questions:
            return "You previously asked:\n" + "\n".join(user_questions)
        else:
            return "You haven't asked any questions yet in this conversation."

# Instantiate and pass memory
recall_tool = RecallChatHistoryTool(conversational_memory=conversational_memory)

query_understander = Agent(
    role="Query Understander",
    goal="Interpret the user's natural language query and extract a structured plan or sequence of intents.",
    backstory=(
        "Expert in understanding semantic intent from user queries, capable of identifying "
        "single queries or decomposing complex multi-step requests into actionable plans."
    ),
    llm=llm,
    verbose=True,
    memory=conversational_memory,
    tools=[IntentMatcherTool, recall_tool], # DefinitionLookup can help it understand terms
)


# Assuming all Tool definitions from previous cell exist above or are imported

# Define the Crew Agent with all tools
mongo_query_planner = Agent(
    role="Mongo Planner",
    goal="Execute MongoDB queries based on the structured plan or intent, potentially chaining multiple tool calls.",
    backstory=(
        "Highly skilled in MongoDB schema, relationships, and efficient query structuring. "
        "Can perform single queries or execute complex, multi-step data retrieval plans, "
        "using results from one step as input for the next."
    ),
    llm=llm,
    verbose=True,
    memory=conversational_memory,
    tools=[
        ComplexQueryExecutorTool,
        DefinitionLookupTool,
        GetAccountLimitTool,
        GetAccountProductsTool,
        GetAccountsByProductTool,
        GetHighLimitAccountsTool,
        ListAllAccountIDsTool,
        GetCustomerByUsernameTool,
        GetCustomersWithEmailDomainTool,
        GetCustomersByAccountTool,
        GetAccountsByCustomerTool,
        GetCustomerTiersTool,
        GetCustomersByBirthYearTool,
        GetAccountsByNameOrUsernameTool,
        GetTransactionsByAccountTool,
        GetTransactionSummaryByTypeTool,
        GetTransactionsBySymbolTool,
        GetTransactionVolumeOverTimeTool,
        FallbackHandlerTool
    ],
    allow_delegation=True
)

result_validator = Agent(
    role="Result Validator",
    goal="Ensure the MongoDB output is meaningful, correctly answers the original query, and is formatted clearly.",
    backstory=(
        "A meticulous interpreter of database responses, refining raw data into "
        "user-friendly and accurate summaries, especially for multi-faceted queries."
    ),
    llm=llm,
    verbose=True,
    memory=conversational_memory
)


query_understanding_task = Task(
    description=(
        "Analyze the user's question: '{user_query}'. "
        "First, use the `IntentMatcherTool` to find a primary intent. "
        "If the query requires multiple steps (e.g., getting customer accounts then transactions for each), "
        "decompose it into a list of sequential operations. "
        "Output a JSON object or list of JSON objects representing the structured intent(s) and their parameters. "
        "Example for single intent: {{'intent':'get_account_limit', 'account_id':758920}}. "
        "Example for multi-step: [{{'action': 'GetAccountsByNameOrUsername', 'parameters': {{'name_or_username': 'John Doe'}}}}, "
        "{{'action': 'GetTransactionSummaryByTypeTool', 'parameters': {{'account_id': '{{PREVIOUS_RESULT.account_ids}}'}}}}] "
        "(Note: '{{PREVIOUS_RESULT.account_ids}}' is a conceptual placeholder for how the planner would consume it)"
    ),
    agent=query_understander,
    expected_output="A JSON structure (object or list) detailing the query intent(s) and parameters.",
    outputs_into_memory=True
)

mongo_execution_task = Task(
    description="Given intent and entities from previous task, run relevant MongoDB action to fetch data.",
    agent=mongo_query_planner,
    expected_output="Raw JSON result from MongoDB.",
    previous_task_memory=True
)

formatting_task = Task(
    description=(
        "Receive the raw MongoDB query results from the 'mongo_execution_task'. "
        "Interpret these results, synthesize them, and format them into a clear, "
        "concise, and user-friendly natural language response. "
        "Ensure the response directly addresses the original user query. "
        "If no relevant data was found, provide a polite informative message."
    ),
    agent=result_validator,
    expected_output="A clear, human-readable summary of the query results, e.g., 'Account 758920 has a limit of $10,000 and products A, B, C. Its transaction summary by type is...'.",
    previous_task_memory=True # This is crucial for accessing previous task's output
)

crew = Crew(
    agents=[query_understander, mongo_query_planner, result_validator],
    tasks=[query_understanding_task, mongo_execution_task, formatting_task],
    memory=True,
    embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="text-embedding-004",
                api_key= os.getenv("GEMINI_API_KEY")
        ),
    verbose=True,
    manager_llm=llm
)
)
def run_crew_agent(user_input: str):
    # Pass the user input to the crew
    crew_output = crew.kickoff(inputs={"user_query": user_input})

    # Extract the final string content from the CrewOutput object
    # The 'raw' attribute typically holds the final human-readable output
    final_response_string = crew_output.raw if hasattr(crew_output, 'raw') else str(crew_output)

    # Save the interaction to conversational memory
    conversational_memory.save_context(
        {"input": user_input},
        {"output": final_response_string} # Pass the extracted string here
    )

    return final_response_string # Return the string to the user