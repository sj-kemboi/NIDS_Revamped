import streamlit as st
import mysql.connector
import pandas as pd
import pickle
import numpy as np
from PIL.Image import Image

# Connect to the database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="38950508",
    database="nids_auth",
    auth_plugin='mysql_native_password'
)

# Create a cursor object
mycursor = mydb.cursor()


# create users table
def users():
    # Create a users table if it does not exist
    mycursor.execute("CREATE TABLE IF NOT EXISTS users "
                     "(id INT AUTO_INCREMENT PRIMARY KEY,"
                     "username VARCHAR(255),"
                     "password VARCHAR(255)")


# adding user information into mydb database - signup
def add_userdata(username, password):
    mycursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                     (username, password))
    mydb.commit()


# login
def login(username, password):
    mycursor.execute("SELECT * FROM users WHERE username = %s AND password = %s",
                     (username, password))
    user = mycursor.fetchone()
    return user



# prediction using realtime data
def nids_dashboard_realtime():
    """
    prediction dashboard, using realtine-data
    """

    # st.write(f'Welcome {name}!')

    st.write('This system helps to detect network intrusions by monitoring network traffic in real-time.')

    # create a text input field for entering the IP address or port number to monitor
    ip_address = st.text_input('Enter the IP address to monitor')

    # create a checkbox to enable/disable real-time monitoring
    real_time = st.checkbox('Enable real-time monitoring')

    # create a button to start/stop monitoring
    if st.button('Start/Stop Monitoring'):
        if real_time:
            # start monitoring network traffic
            st.write('Monitoring network traffic...')
            # code to monitor network traffic in real-time
        else:
            # stop monitoring network traffic
            st.write('Stopped monitoring network traffic.')


def main():
    global result

    # Initialize session state ==> holds the logged_in state
    if 'key' not in st.session_state:
        st.session_state.key = False

    # Initialize session state ==> holds the username
    if 'username_state' not in st.session_state:
        st.session_state.username_state = ""

    # Sidebar menu
    if st.session_state.key:
        menu = ["Dataset", "Real-time Data", "Logout"]
    else:
        menu = ["Home", "Login", "Signup"]

    # menu = ["Home", "Login", "Signup", "Logout"]
    option = st.sidebar.selectbox("Select an option", menu)

    # if user is logged in
    if st.session_state.key:

        if option == "Dataset":

            st.title(f"Hello, {st.session_state.username_state}")

            # create a title and description for the app
            st.subheader('Detect Intrusion using a Dataset')
            st.write('This system helps to detect network intrusions by monitoring'
                     ' network traffic on provided data.')

            # Upload a dataset
            dataset = st.file_uploader("Upload your Dataset", type=["csv"])

            # traffic_data = []
            if dataset is not None:
                # load the dataset
                traffic_data = pd.read_csv('Test_data.csv')

                # drop redundant columns
                traffic_data.drop(['num_outbound_cmds'], axis=1, inplace=True)

                # feature scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()

                cols = traffic_data.select_dtypes(include=['int64', 'float64']).columns
                sc_test = scaler.fit_transform(traffic_data.select_dtypes(include=['int64', 'float64']))
                std_test = pd.DataFrame(sc_test, columns=cols)

                # encoding categorical values
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()

                cattest = traffic_data.select_dtypes(include=['object']).copy()
                en_test = cattest.apply(encoder.fit_transform)

                # join prerocessed and encoded data
                en_test_data = pd.concat([std_test, en_test], axis=1)

                test_data = en_test_data

                st.write('Network Traffic Data:')
                st.table(en_test_data.head())

                # loading the model (trainned_model.pkl)
                model = pickle.load(open('trainned_model.pkl', 'rb'))

                # predicting the dataset using the model
                prediction = model.predict(test_data)

                num_rows = test_data.shape[0]
                st.write(f"Total number of rows :: {num_rows}")

                # user to input the number of rows to be predicted
                user_input = st.number_input("Number of rows to be predicted : ", value=0, step=1)

                # Testing for random rows
                random_rows = np.random.randint(len(test_data), size=user_input)

                if st.button("Predict Data"):

                    for j in random_rows:
                        st.write("For row : ", j)
                        predicted_value = prediction.reshape(1, -1)[0][j]
                        st.write(f'Predicted value :  {predicted_value}')
                        if predicted_value == 1:
                            st.write(f"Possible intrusion at row : {j}")
                        else:
                            st.write(f"Normal network at row : {j}")
                        st.write("\n")

            else:
                st.error("Import a dataset")

        elif option == "Real-time Data":

            st.title(f"Hello, {st.session_state.username_state}")

            st.subheader('Detect Intrusion using Real-time Data')

            st.write('This system helps to detect network intrusions by monitoring'
                     ' network traffic in real-time.')

            # create a text input field for entering the IP address or port number to monitor
            ip_address = st.text_input('Enter the IP address to monitor')

            # create a checkbox to enable/disable real-time monitoring
            real_time = st.checkbox('Enable real-time monitoring')

            # create a button to start/stop monitoring
            if st.button('Start/Stop Monitoring'):
                if real_time:
                    # start monitoring network traffic
                    st.write('Monitoring network traffic...')
                    # code to monitor network traffic in real-time
                else:
                    # stop monitoring network traffic
                    st.write('Stopped monitoring network traffic.')

        elif option == "Logout":
            # Create a widget to update session state:
            st.session_state.key = False
            # st.experimental_rerun()

    # if user is not logged in
    else:
        if option == "Home":

            # Set page title and layout
            st.title('Network Intrusion Detection')

            # load a background image
            st.image("network1.png", width=500)

            st.subheader("Welcome")

            st.write("Login to access the dashboard and monitor your network.")

        # Login page
        elif option == "Login":
            st.title("NIDS Authentication")
            st.write("Please enter your credentials to login to the NIDS dashboard.")
            st.subheader('Login')
            form = st.form(key="login_form", clear_on_submit=True)
            username_input = form.text_input("Username", placeholder='Enter username')
            password_input = form.text_input("Password", placeholder='Enter password', type="password")
            submit_button = form.form_submit_button("Login")

            if submit_button:
                result = login(username_input, password_input)

                if result:

                    st.session_state.username_state = username_input
                    st.success(f"Welcome {st.session_state.username_state}")

                    # Create a widget to update session state:
                    st.session_state.key = result
                    st.experimental_rerun()

                else:
                    st.error("User does not exist. Please Signup!")

        elif option == "Signup":
            st.title("NIDS Authentication")
            st.write("Please create a new account to access the NIDS dashboard.")
            st.subheader('Sign Up')
            form = st.form(key="signup_form", clear_on_submit=True)
            username_input = form.text_input("Username", placeholder='Enter username')
            password_input = form.text_input("Password", placeholder='Enter password', type="password")
            confirm_password_input = form.text_input("Confirm Password", placeholder='Confirm password',
                                                     type="password")
            submit_button = form.form_submit_button("Signup")

            # Verify user credentials

            if submit_button:
                mycursor.execute("SELECT * FROM users WHERE username = %s AND password = %s",
                                 (username_input, password_input))
                user = mycursor.fetchone()
                if user:
                    st.error("User already exists.")
                else:
                    if len(username_input) >= 2:
                        if not username_input.isnumeric():
                            if username_input != "":
                                if password_input == confirm_password_input:
                                    if len(password_input) >= 8:
                                        if password_input != "":
                                            if not password_input.isnumeric():
                                                # adding user info into the db
                                                add_userdata(username_input, password_input)
                                                st.success("Account created successfully."
                                                           "Please login to access the NIDS dashboard.")
                                            else:
                                                st.error("Password should contain at least one character")
                                        else:
                                            st.error("Password should not be blank")
                                    else:
                                        st.error("Password should more than 8 Characters")
                                else:
                                    st.error("Password don't match")
                            else:
                                st.error("Username should not be blank")
                        else:
                            st.error("Username cannot contain digits only")
                    else:
                        st.error("Username should contain 2 or more characters")

        else:
            st.error("Invalid username or password. Please try again.")


if __name__ == '__main__':
    main()
