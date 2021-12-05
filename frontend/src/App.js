import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

{/* Can be used to check user in db and also register */}
function LoginPostQuery(query_email,query_password,end){
  var myParams = {
    email: query_email,
    password: query_password
}

var fullurl= 'http://127.0.0.1:5000' + end
  const [getMessage, setGetMessage] = useState({})
  useEffect(()=>{
    axios.post(fullurl, myParams).then(response => {
      console.log("SUCCESS", response)
      setGetMessage(response.data)
    }).catch(error => {
      console.log(error)
    })
  }, [])
  return(getMessage)
}
{/* get user Risk */}
function PortRiskPostQuery(risk,end){
  var myParams = {
    port_risk: risk
}

var fullurl= 'http://127.0.0.1:5000' + end
  const [getMessage, setGetMessage] = useState({})
  useEffect(()=>{
    axios.post(fullurl, myParams).then(response => {
      console.log("SUCCESS", response)
      setGetMessage(response.data)
    }).catch(error => {
      console.log(error)
    })
  }, [])
  return(getMessage)
}
{/* new user: input risk to create portfolio*/}
function NewUserPortfolioPostQuery(query_risk, query_email, end){
  var myParams = {
    email: query_email,
    risk_appetite: query_risk
}

var fullurl= 'http://127.0.0.1:5000' + end
  const [getMessage, setGetMessage] = useState({})
  useEffect(()=>{
    axios.post(fullurl, myParams).then(response => {
      console.log("SUCCESS", response)
      setGetMessage(response.data)
    }).catch(error => {
      console.log(error)
    })
  }, [])
  return(getMessage)
}

{/* get portfolio ID*/}
function Get_Portfolio_ID(query_email){
  const [getMessage, setGetMessage] = useState({})
  var full_url = 'http://127.0.0.1:5000/portfolios/pushPortfolioid/' + query_email
  useEffect(()=>{
    axios.get(full_url).then(response => {
      console.log("SUCCESS", response)
      setGetMessage(response)
    }).catch(error => {
      console.log(error)
    })
  }, [])
  return(getMessage)
}
function App() {
  const [getMessage, setGetMessage] = useState({})

{/* sign in: need username and password */}
 const outs= LoginPostQuery('jefftsai1999@gmail.com','jefftsai','/users/login')

{/* register: need username and password */}
{/*const outs2= LoginPostQuery('jefftsai1999@gmail.com','jefftsais','/users/register')*/}

{/* if new user, input user params to create new portfolio: need risk (high,medium,low) and email for now*/}
{/*const outs3 = Get_Portfolio_ID('jefftsai1999@gmail.com')*/}
{/*const outs3 = NewUserPortfolioPostQuery('high', 'jefftsai1998@gmail.com', 'portfolios/new') */}

{/* if old user, get portfolio and input user params, need email and portfolio id */}


  return (
    <div className="App">
      <header className="App-header">
        {outs.Status}
        {"             "  }

        <img src={logo} className="App-logo" alt="logo" />
      </header>
    </div>
  );
}

export default App;
