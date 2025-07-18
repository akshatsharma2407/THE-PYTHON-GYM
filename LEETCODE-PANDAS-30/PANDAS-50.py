# ============================================================================================================================================================================================================

# A country is big if:
# it has an area of at least three million (i.e., 3000000 km2), or
# it has a population of at least twenty-five million (i.e., 25000000).
# Write a solution to find the name, population, and area of the big countries.
def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    df = world[(world['area'] >= 3000000) | (world['population'] >= 25000000)]
    return df[['name','population','area']]





# Write a solution to find the ids of products that are both low fat and recyclable.
# Return the result table in any order.
def find_products(products: pd.DataFrame) -> pd.DataFrame:
    x = products[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y')]['product_id']
    return pd.DataFrame(x)





# Write a solution to find all customers who never order anything.
# Return the result table in any order.
def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(customers[~customers['id'].isin(orders['customerId'])]['name']).rename(columns = {'name' : 'Customers'})





# Write a solution to find all the authors that viewed at least one of their own articles.
# Return the result table sorted by id in ascending order.
def article_views(views: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(views[views['author_id'] == views['viewer_id']]['author_id'].sort_values().unique())
    return x.rename(columns = {0 : 'id'})





# Write a solution to find the IDs of the invalid tweets. The tweet is invalid if the number of characters used in the content of the tweet is strictly greater than 15.
def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
   tweets['lens'] = tweets['content'].str.len()
   return pd.DataFrame(tweets.query('lens > 15')['tweet_id'])





# Write a solution to calculate the bonus of each employee. The bonus of an employee is 100% of their salary if the ID of the employee is an odd number and the employee's name does not start with the character 'M'. The bonus of an employee is 0 otherwise.
# Return the result table ordered by employee_id.
def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus'] = employees.apply(lambda x : x['salary'] if (x['employee_id'] % 2 != 0) & (x['name'][0] != 'M') else 0, axis = 1)
    employees.sort_values('employee_id',inplace=True)
    return employees[['employee_id','bonus']]





# Write a solution to fix the names so that only the first character is uppercase and the rest are lowercase.
# Return the result table ordered by user_id.
def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    users['name'] = users['name'].apply(lambda x : x.capitalize())
    users = users.sort_values('user_id')
    return users





# Write a solution to find the users who have valid emails.
# A valid e-mail has a prefix name and a domain where:
# The prefix name is a string that may contain letters (upper or lower case), digits, underscore '_', period '.', and/or dash '-'. The prefix name must start with a letter.
# The domain is '@leetcode.com'.
def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    return users[users['mail'].str.match(r'^[A-Za-z][A-Za-z0-9_\.\-]*@leetcode\.com$')]





# Write a solution to find the patient_id, patient_name, and conditions of the patients who have Type I Diabetes. Type I Diabetes always starts with DIAB1 prefix.
# Return the result table in any order.
def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    bl = []
    for i in patients['conditions']:
        l = i.split(' ')

        flag = False
        for j in l:
            if j.startswith('DIAB1'):
                flag = True
                bl.append(flag)
                continue
        
        if flag == False:
            bl.append(flag)
    
    patients['cnd'] = bl

    return patients[patients['cnd'] == True][['patient_id','patient_name','conditions']]





# Write a solution to find the nth highest salary from the Employee table. If there is no nth highest salary, return null.
# The result format is in the following example.
def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    employee['rank'] = employee['salary'].rank(ascending=False,method='dense')
    x = employee[employee['rank'] == N]['salary'].values
    if x.shape[0] == 0:
        return pd.DataFrame([[None]],columns=[f'getNthHighestSalary({N})'])
    elif x.shape[0] > 1:
        return pd.DataFrame([[x[0]]],columns=[f'getNthHighestSalary({N})'])
    return pd.DataFrame(x,columns=[f'getNthHighestSalary({N})'])





# Write a solution to find the second highest distinct salary from the Employee table. If there is no second highest salary, return null (return None in Pandas).
def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    employee.drop_duplicates(subset='salary',inplace=True)
    employee['rank'] = employee['salary'].rank(ascending=False)
    x = employee[employee['rank'] == 2]['salary'].values
    if x.shape[0] == 0:
        return pd.DataFrame([[None]],columns=['SecondHighestSalary'])
    x = x[0]
    return pd.DataFrame([[x]],columns=['SecondHighestSalary'])





# Write a solution to find employees who have the highest salary in each of the departments.
def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    employee['ranks'] = employee.groupby('departmentId')['salary'].rank(method='dense',ascending=False)
    employee = employee[employee['ranks'] == 1.0]
    df = employee.merge(department,left_on = 'departmentId' , right_on = 'id')
    return df[['name_y','name_x','salary']].rename(
        columns =
        {
            'name_y' : 'Department',
            'name_x' : 'Employee'
        }
    )





# Write a solution to find the rank of the scores. The ranking should be calculated according to the following rules:
# The scores should be ranked from the highest to the lowest.
# If there is a tie between two scores, both should have the same ranking.
# After a tie, the next ranking number should be the next consecutive integer value. In other words, there should be no holes between ranks.
def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    scores['rank'] = scores['score'].rank(method='dense',ascending=False)
    df = scores.sort_values(by='score',ascending=False)
    return df[['score','rank']]





# Write a solution to delete all duplicate emails, keeping only one unique email with the smallest id.
# For Pandas users, please note that you are supposed to modify Person in place.
# After running your script, the answer shown is the Person table. The driver will first compile and run your piece of code and then show the Person table. The final order of the Person table does not matter.
def delete_duplicate_emails(person: pd.DataFrame) -> None:
    person.sort_values('id',inplace=True)
    person.drop_duplicates(subset='email',inplace=True)





# Write a solution to rearrange the Products table so that each row has (product_id, store, price). If a product is not available in a store, do not include a row with that product_id and store combination in the result table.
# Return the result table in any order.
def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:
    df = products.melt(id_vars = ['product_id'],var_name='store',value_name='price')
    df.dropna(inplace=True)
    df.sort_values(['product_id','price'],inplace=True)
    return df





# Write a solution to calculate the number of bank accounts for each salary category. The salary categories are:
# "Low Salary": All the salaries strictly less than $20000.
# "Average Salary": All the salaries in the inclusive range [$20000, $50000].
# "High Salary": All the salaries strictly greater than $50000.
# The result table must contain all three categories. If there are no accounts in a category, return 0.
# Return the result table in any order.
def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    category = pd.DataFrame(['Low Salary','Average Salary','High Salary'],columns = ['cat'])

    cat = []
    for i in accounts['income']:
        if i < 20000:
            cat.append('Low Salary')
        elif i <= 50000:
            cat.append('Average Salary')
        else:
            cat.append('High Salary')
    
    accounts['cat'] = cat

    df = accounts.merge(category,how='outer',on='cat')

    df.reset_index(inplace=True)

    index = []

    for i in df['income']:
        if type(i) == pd._libs.missing.NAType: #got this term by error while solving this problem
            index.append(0)
        else:
            index.append(1)
    
    df['index'] = index
    
    df = df.groupby('cat')['index'].sum().reset_index()
    
    return df.rename(
        columns = 
        {
            'cat' : 'category',
            'index' : 'accounts_count'
        }
    )





# Write a solution to calculate the total time in minutes spent by each employee on each day at the office. Note that within one day, an employee can enter and leave more than once. The time spent in the office for a single entry is out_time - in_time.
# Return the result table in any order.
def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    employees['time'] = employees['out_time'] - employees['in_time']
    df = employees.groupby(['event_day','emp_id']).agg(
        {
            'time' : 'sum'
        }
    ).reset_index()

    return df.rename(columns=
    {
        'event_day' : 'day',
        'time' : 'total_time'
    })





# Write a solution to find the first login date for each player.
# Return the result table in any order.
def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    return activity.groupby('player_id')['event_date'].min().reset_index().rename(
        columns = {
            'event_date' : 'first_login'
        }
    )

# Write a solution to calculate the number of unique subjects each teacher teaches in the university.
# Return the result table in any order.
def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
    return teacher.groupby('teacher_id')['subject_id'].nunique().reset_index().rename(
        columns =
        {
            'subject_id' : 'cnt'
        }
    )






# Write a solution to find all the classes that have at least five students.
# Return the result table in any order.
def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    df = courses.groupby('class')['student'].count().reset_index()
    return pd.DataFrame(df[df['student'] > 4]['class'])





# Write a solution to find the customer_number for the customer who has placed the largest number of orders.
# The test cases are generated so that exactly one customer will have placed more orders than any other customer.
def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(orders.groupby('customer_number')['order_number'].count().sort_values(ascending=False).reset_index().head(1)['customer_number'])






# Write a solution to find for each date the number of different products sold and their names.
# The sold products names for each date should be sorted lexicographically.
# Return the result table ordered by sell_date.
def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    df1 = activities.groupby('sell_date').agg(
        {
            'product' : 'unique'
        }
    ).reset_index()

    df1['product'] = df1['product'].apply(lambda x : ','.join(sorted(x)))

    df2 = activities.groupby('sell_date').agg(
        {
            'product' : 'nunique'
        }
    ).reset_index()

    return df2.merge(df1,on='sell_date').rename(
        columns = 
        {
            'product_x' : 'num_sold',
            'product_y' : 'products'
        }
    )





# For each date_id and make_name, find the number of distinct lead_id's and distinct partner_id's.
# Return the result table in any order.
def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    return daily_sales.groupby(['date_id','make_name']).agg(
        {
            'lead_id' : 'nunique',
            'partner_id' : 'nunique'
        }
    ).reset_index().rename(
        columns = 
        {
            'lead_id' : 'unique_leads',
            'partner_id' : 'unique_partners'
        }
    )





# Write a solution to find all the pairs (actor_id, director_id) where the actor has cooperated with the director at least three times.
# Return the result table in any order.
def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    return actor_director.groupby(['actor_id','director_id'])['timestamp'].count().reset_index().query('timestamp > 2')[['actor_id','director_id']]






# Write a solution to show the unique ID of each user, If a user does not have a unique ID replace just show null.
# Return the result table in any order.
def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    return employees.merge(employee_uni,how='left',on='id')[['unique_id','name']]




# Write a solution to find the number of times each student attended each exam.
# Return the result table ordered by student_id and subject_name.
def students_and_examinations(students: pd.DataFrame, subjects: pd.DataFrame, examinations: pd.DataFrame) -> pd.DataFrame:
    examinations = examinations.reset_index()
    exam_cnt = examinations.groupby(['student_id','subject_name'])['index'].count().reset_index()

    st_sj = students.merge(subjects,how='cross')

    df = exam_cnt.merge(st_sj,how='right',on=['student_id','subject_name'])

    df['index'] = df['index'].fillna(0)


    return df[['student_id','student_name','subject_name','index']].rename(
        columns = {
            'index' : 'attended_exams'
        }
    ).sort_values(by=['student_id','subject_name'])






# Write a solution to find managers with at least five direct reports.
# Return the result table in any order.
def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
   return pd.DataFrame(employee[employee['id'].isin(employee.groupby(['managerId'])['id'].count().reset_index().query('id > 4')['managerId'].values)]['name'])





# Write a solution to find the names of all the salespersons who did not have any orders related to the company with the name "RED".
# Return the result table in any order.
def sales_person(sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(sales_person[~sales_person.isin(orders[orders['com_id'].isin(company[company['name'] == 'RED']['com_id'].values)]['sales_id'].values)][['sales_id','name']].dropna()['name'])






# ======================================================================================END=====================================================================================
