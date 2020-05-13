### Intro to Deployment
![](https://github.com/armandordorica/Advanced-Python/blob/master/intro_to_deployment/machine_learning_workflow.png?raw=true)

**Explore and process data**
1. Retrieve - getting the raw data 
2. Clean and explore - Explore and visualize the data to identify the most prominent features. 
    * Remove outliers or mistakes
3. Prepare/transform
    * Most Machine Learning models expect standardized data values. Therefore, this step often involves normalizing and converting the format of the data. 
    * In addition, split the data into training, validation, and test datasets. 
  
  **Modeling**
  1. Develop and train the model using the training dataset. 
  2. Final step of modeling is to evaluate and validate the model. In this step, you'll tune the model using the validation dataset. 
  3. Finally, evaluate the model by using the test dataset. 
  
  **Deployment**
  
  Deployment, monitoring, and updating the model in the production environment.
  1. Deployment
      * Make your model available for use by web or software application. 
  2. Monitor and update model and data
  
  
### Prod environment schematic 
![](https://github.com/armandordorica/Advanced-Python/blob/master/intro_to_deployment/prod_env_schematic.png?raw=true)

### Endpoints and REST APIs 
![](https://github.com/armandordorica/Advanced-Python/blob/master/intro_to_deployment/application_endpoint_model_diagram.png?raw=true)

* Endpoint is the interface to the model, which facilitates an ease of communication between the model and the application. 
* The endpoint allows to SEND user data to the model and RECEIVES predictions back from the model based upon that user data. 
* Accepts user data as the input and returns the model's prediction based upon this input through the endpoint. 
* The endpoint is an API, which uses REpresentational State Transfer (REST) architecture, which is a framework for the set of rules and constraints that must be adhered to for communication between programs. 

#### How this translates to Python
* Endpoint --> function call
* Model  --> function itself 
* Application --> Python program

#### Rest API
* Uses HTTP request and responses to allow communication between the application and the model via the endpoint (interface). 

**Parts of the HTTP request from application --> model**
1. Endpoint
   * It's a URL (Uniform Resource Locator), which is commonly known as a web address.This enabbles communication between the application and the model using a REST API. 
2. HTTP method 
   * The main HTTP methods are GET, POST, PUT and DELETE but for deployment purposes we only care about the POST method. 
3. HTTP headers
   * The headers contain additional information, like the format of the data within the message, that's past to the *receiving* program 
4. Message (Data or Body) 
   * The final part is the message (data or body); for deployment will contain the *user's data* which is the input into the model. 
   
**Parts of the HTTP request from model --> application**
1. HTTP status code 
   * If the model successfully received and processed the *user's data* that was sent in the message, it will return an OK status that starts with a 2, i.e. `STATUS 200`. 
2. HTTP headers 
   * They contain additional information, like the format of the data within the message that is passed to the receiving program. 
3. Message (Data or Body)
   * The data returned within the message is the *prediction* as returned by the model. This prediction is then passed to the application user via the application. 

   




* What's the machine learning workflow?

* How does deployment fit into the machine learning workflow?

* What is cloud computing?

* Why would we use cloud computing for deploying machine learning models?

* Why isn't deployment a part of many machine learning curriculums?

* What does it mean for a model to be deployed?

* What are the essential characteristics associated with the code of deployed models?

* What are different cloud computing platforms we might use to deploy our machine learning models?
