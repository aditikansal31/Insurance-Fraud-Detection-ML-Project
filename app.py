from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("Random_Forest_Model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


policy_state_map = {
    "IL": 0,
    "IN": 1,
    "OH": 2
}

policy_csl_map = {
    "250/500": 0,
    "100/300": 1,
    "500/1000": 2
}

education_map = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4
}

incident_type_map = {
    "Single Vehicle Collision": 0,
    "Vehicle Theft": 1,
    "Parked Car": 2,
    "Multi-vehicle Collision": 3
}

collision_type_map = {
    "Front Collision": 0,
    "Rear Collision": 1,
    "Side Collision": 2
}

severity_map = {
    "Minor Damage": 0,
    "Major Damage": 1,
    "Total Loss": 2,
    "Trivial Damage": 3
}

yes_no_map = {
    "NO": 1,
    "YES": 2
}


@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    policy_number = float(request.form['policy_number'])

    policy_state = policy_state_map[request.form['policy_state']]
    policy_csl = policy_csl_map[request.form['policy_csl']]

    policy_deductable = float(request.form['policy_deductable'])
    policy_annual_premium = float(request.form['policy_annual_premium'])
    umbrella_limit = float(request.form['umbrella_limit'])
    insured_zip = float(request.form['insured_zip'])
    insured_sex = float(request.form['insured_sex'])

    insured_education_level = education_map[request.form['insured_education_level']]

    insured_occupation = float(request.form['insured_occupation'])
    insured_hobbies = float(request.form['insured_hobbies'])
    insured_relationship = float(request.form['insured_relationship'])

    capital_gains = float(request.form['capital-gains'])
    capital_loss = float(request.form['capital-loss'])

    incident_type = incident_type_map[request.form['incident_type']]
    collision_type = collision_type_map[request.form['collision_type']]
    incident_severity = severity_map[request.form['incident_severity']]

    authorities_contacted = float(request.form['authorities_contacted'])
    incident_state = float(request.form['incident_state'])
    incident_city = float(request.form['incident_city'])
    incident_location = float(request.form['incident_location'])
    incident_hour = float(request.form['incident_hour_of_the_day'])

    vehicles_involved = float(request.form['number_of_vehicles_involved'])

    property_damage = yes_no_map[request.form['property_damage']]

    bodily_injuries = float(request.form['bodily_injuries'])
    witnesses = float(request.form['witnesses'])

    police_report_available = yes_no_map[request.form['police_report_available']]

    total_claim_amount = float(request.form['total_claim_amount'])
    auto_make = float(request.form['auto_make'])
    auto_model = float(request.form['auto_model'])
    auto_year = float(request.form['auto_year'])

    policy_annual_premium_log = float(request.form['policy_annual_premium_log'])
    claim_delay = float(request.form['claim_delay'])

    features = [[
        age,
        policy_number,
        policy_state,
        policy_csl,
        policy_deductable,
        policy_annual_premium,
        umbrella_limit,
        insured_zip,
        insured_sex,
        insured_education_level,
        insured_occupation,
        insured_hobbies,
        insured_relationship,
        capital_gains,
        capital_loss,
        incident_type,
        collision_type,
        incident_severity,
        authorities_contacted,
        incident_state,
        incident_city,
        incident_location,
        incident_hour,
        vehicles_involved,
        property_damage,
        bodily_injuries,
        witnesses,
        police_report_available,
        total_claim_amount,
        auto_make,
        auto_model,
        auto_year,
        policy_annual_premium_log,
        claim_delay
    ]]

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Fraudulent Insurance Claim"
    else:
        result = "Legitimate Insurance Claim"

    return render_template("result.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)