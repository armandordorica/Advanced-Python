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

   

### Model, Application and Containers 
Production environment is mainly compoed of two programs: **model** and **application** that communicate with each other through the **endpoint** (interface). 

Th model and the application require a computing environment so that they can be run and be available for use. 
* **Containers** allow to create and maintain these computing environmnets. Containers are created with a script that specifies which software packages, libraries, and other computer attributes are necessary to run the software application, in our case either the **model** or the **application**. 
* The structure of a **Docker container** enables to create, save, use and delete a set of common tools. 


### Docker Containers 
* A container script file is used to create a container.
* This text script file can easily be shared with others and provides a simple method to replicate a particular container.
* This container script is simply the instructions (algorithm) that is used to create a container; for Docker these container scripts are referred to as dockerfiles.
* Container script files can be stored in repositories, which provide a simple means to share and replicate containers

**Container Structure**
* Computational infrastructure - cloud, on-prem server, local computer. 
* Operating System
* Container Engine - Docker software running on the computer that enables to create, save, use, and delete containers. 

**Architecture of containers provides the following advantages:**

* Isolates applciation --> Increased security. 
* Only software to run the application is required --> More efficient use of resources and faster application deployment.
* Makes application creation, replication, deletion, and maintenance easier and consistent across applications that are deployed using containers. 
* Simpler and more secure way to replicate, save, and share containers. 


* What's the machine learning workflow?

* How does deployment fit into the machine learning workflow?

* What is cloud computing?

* Why would we use cloud computing for deploying machine learning models?

* Why isn't deployment a part of many machine learning curriculums?

* What does it mean for a model to be deployed?

* What are the essential characteristics associated with the code of deployed models?

* What are different cloud computing platforms we might use to deploy our machine learning models?



### Paths to Deployment 
**Paths to Deployment:**
1. Python model is recoded into the programming language of the production environment.
   * The first method which involves recoding the Python model into the language of the production environment, often Java or C++. This method is rarely used anymore because it takes time to recode, test, and validate the model that provides the same predictions as the original.
2. Model is coded in Predictive Model Markup Language (PMML) or Portable Format Analytics (PFA).
   * The second method is to code the model in Predictive Model Markup Language (PMML) or Portable Format for Analytics (PFA), which are two complementary standards that simplify moving predictive models to deployment into a production environment. The Data Mining Group developed both PMML and PFA to provide vendor-neutral executable model specifications for certain predictive models used by data mining and machine learning. Certain analytic software allows for the direct import of PMML including but not limited to IBM SPSS, R, SAS Base & Enterprise Miner, Apache Spark, Teradata Warehouse Miner, and TIBCO Spotfire.


3. Python model is converted into a format that can be used in the production environment.

* The third method is to **build a Python model and use libraries and methods that convert the model into code that can be used in the production environment**. Specifically most popular machine learning software frameworks, like PyTorch, TensorFlow, SciKit-Learn, have methods that will convert Python models into intermediate standard format, like ONNX (Open Neural Network Exchange format). This intermediate standard format then can be converted into the software native to the production environment.

This is the easiest and fastest way to move a Python model from modeling directly to deployment.
Moving forward this is typically the way models are moved into the production environment.
Technologies like containers, endpoints, and APIs (Application Programming Interfaces) also help ease the work required for deploying a model into the production environment.


### DevOps Diagram 
![](https://github.com/armandordorica/Advanced-Python/blob/master/devOps_diagram.png?raw=true)


### Deploying a Model with SageMaker 
![](https://github.com/armandordorica/Advanced-Python/blob/master/intro_to_deployment/intro_to_deployment:endpoints.png?raw=true)

An endpoint, in this case, is a URL that allows an application and a model to speak to one another.
* You can start an endpoint by calling `.deploy()` on an estimator and passing in some information about the instance.

```
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')
```

* Then, you need to tell your endpoint, what type of data it expects to see as input (like .csv).
```
from sagemaker.predictor import csv_serializer

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
```

* Then, perform inference; you can pass some data as the "Body" of a message, to an endpoint and get a response back. 
```
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint,   # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))
```
* The inference data is stored in the "Body" of the response, and can be retrieved:
```
response = response['Body'].read().decode('utf-8')
print(response)
```

* Finally, do not forget to shut down your endpoint when you are done using it.
```
xgb_predictor.delete_endpoint()
```

