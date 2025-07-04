from tools import query_policy, employee_data_query

def simple_agent(query: str) -> str:
    if "joining date" in query.lower() or "employee" in query.lower():
        return employee_data_query(query)
    else:
        return query_policy(query)
