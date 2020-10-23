import numpy as np
import random
import turtle


turtle.bgcolor("black")
#This function creates the neural network and coordinates of the bugs
def initialize_intellect_coordinates(length_of_intellect,random_seed,data_length,num_data):#length of the food. 
    random.seed(random_seed)
    
    intellect = {}
    intellect["coordinates"] = np.ones((4,1))
    intellect["W1"] = np.random.randn(length_of_intellect,data_length)#B
    intellect["b1"] = np.zeros((length_of_intellect,1))
    intellect["W2"] = np.random.randn(4,length_of_intellect)
    intellect["b2"] = np.zeros((4,1))
    intellect["decider_W3"] = np.random.randn(1,num_data)
    intellect["decider_b3"] = np.zeros((1,1))
    intellect["score"] = 0
    intellect["distance_list"] = []
    return intellect 
 
#This function creates as many bugs as we want.
def make_bugs(random_seed,num_bugs,length_of_intellect,data_length,num_data):
    random.seed(3)
    bugs = {}
    for i in range(0,num_bugs):
        bugs["bug" + str(i)] = initialize_intellect_coordinates(length_of_intellect,i,data_length,num_data)
        
    return bugs
    

def make_bugs_visual(random_seed,num_bugs,length_of_intellect,data_length,num_data):#E
    colors = ["blue","yellow","red","purple","orange"]
    
    random.seed(3)
    bugs = {}
    turtles = {}
    for i in range(0,num_bugs):
        bugs["bug" + str(i)] = initialize_intellect_coordinates(length_of_intellect,i,data_length,num_data)
        turtles["bug" + str(i)] = turtle.Turtle()
        turtles["bug" + str(i)].shapesize(1,1,1)
        turtles["bug" + str(i)].shape("turtle")
        turtles["bug" + str(i)].color(random.choice(colors))
        turtles["bug" + str(i)].pensize(1)
        turtles["bug" + str(i)].speed(5)
        turtles["bug" + str(i)].penup()
        turtles["bug" + str(i)].setposition(1,1)
    return bugs,turtles
 
#This function give us the inherent features of food and the coordinates of the food.
def initialize_food(random_seed):
    random.seed(random_seed)
    food_features = {}
    food_features["coordinates"] = np.random.randn(4,1)*20 
    food_features["chemical_volatility"] = np.random.randn(4,1)#D
    #food_features["merged"] = np.append(food_features["coordinates"] ,food_features["chemical_volatility"])
    return food_features["coordinates"]#food_features["merged"]
 
#This fuınction create foods as many as we want
def make_food(num_food):
    random.seed(4)
    food = {}
    for f in range(0,num_food):
        food["fo" + str(f)] = initialize_food(f)
    return food



def make_food_visual(num_food):
    random.seed(4)
    food = {}
    food_turt = {}
    for f in range(0,num_food):
        food["fo" + str(f)] = initialize_food(f)
        food_turt["fo" + str(f)] = turtle.Turtle()
        food_turt["fo" + str(f)].shapesize(1,1,1)
        food_turt["fo" + str(f)].shape("circle")
        food_turt["fo" + str(f)].color("red")
        food_turt["fo" + str(f)].penup()
        coordo = food["fo" + str(f)]
        coordo.reshape(coordo.shape[0],1)
        print(coordo)
        food_turt["fo" + str(f)].setposition(int(coordo[0])-int(coordo[3]),int(coordo[1])-int(coordo[2]))
        print(int(coordo[0])-int(coordo[3]),int(coordo[1])-int(coordo[2]))
    return food,food_turt
 
#This function describes the act of a bug thinking of which route to take. Which is a simple forward propagation
def bug_deciding(bug_name,bugs,food):
    bugy = bugs[str(bug_name)]["coordinates"]
    W1 = bugs[str(bug_name)]["W1"]
    b1 = bugs[str(bug_name)]["b1"]
    W2 = bugs[str(bug_name)]["W2"]
    b2 = bugs[str(bug_name)]["b2"]
    decider_W3 = bugs[str(bug_name)]["decider_W3"]
    decider_b3 = bugs[str(bug_name)]["decider_b3"]#İ
    #at this step we also add the coordinates of the bug to the data
    a = food["fo" + str(0)]
    a = a.reshape(a.shape[0],1)
    #print("a is :",a)
    for i in range(1,len(food)):
        b = food["fo"+ str(i)]
        b = b.reshape(b.shape[0],1)
        #print("b is:" , b)
        a = np.append(a,b,axis = 1)
    
    k = np.ones([a.shape[0],a.shape[1]])
    #print("k is: " , k)
    second = k * bugy
    #print("second is : ",second)
    a = np.append(a,second,axis = 0)    
    
    #print("result : ",a)
   
    #this part is the forward propagation part
    
    Z1 = np.dot(W1,a) + b1
    A1 = Z1
    Z2 = np.dot(W2,A1) + b2
    A2 = Z2
    Z3 = np.dot(decider_W3,A2.T) + decider_b3#R
    AL = Z3
    #so as a result bug looks at all of the data. And returns an array the size(4,#data)
    
    #we will make a single layer neural network
    return AL.T
    
#This function generalizes deciding_act to all the bugs
def all_the_bugs_deciding(bugs,food):
    movement_points = {}
    for k in range(len(bugs)):
        movement_points["bug" + str(k)] = bug_deciding("bug" + str(k) , bugs,food)
        
    return movement_points
        
 
    
#This function let's bugs do their ,thought moves.    
def move_all_the_bugs(bugs,movement_points):
    for i in range(len(bugs)):
        bugs["bug" + str(i)]["coordinates"] = bugs["bug" + str(i)]["coordinates"] +  movement_points["bug" + str(i)]
    return bugs
  
def evaluating_bugs(bugs,food):
#take a bug and calculate it s distance from all the food. 
    
    for i in range(len(bugs)):
        distance_list = bugs["bug" + str(i)]["distance_list"]
        for j in range(len(food)):
            vector_dif = bugs["bug" + str(i)]["coordinates"] - food["fo" + str(j)]
            distance = (np.dot(vector_dif.T,vector_dif))** 0.5
            distance_list.append(distance)
        bugs["bug" + str(i)]["score"] = min(distance_list)
        bugs["bug" + str(i)]["distance_list"] = []
    
            
        
        
def natural_selection(bugs,food,surv_num = 4):#The top two bugs that get closest to the food gives birth to next generation.
#This function will calculate distance of the bugs to the closest food resources and the closest ones are rewarded with reproduction rights.
#get the coordinate of bug 1 , find distances for all the food,
    evaluating_bugs(bugs,food)
    
    bug_score = {}
    
    for i in range(len(bugs)):
    
        bug_score["bug" + str(i)] = bugs["bug" + str(i)]["score"]
        
    score_sorted = sorted(bug_score.items(), key = lambda x:x[1],reverse=False)
    
    survivor_num = len(bugs) // surv_num
    
    survivors = score_sorted[0:survivor_num]
    survivor_names = []
    for s in range(len(survivors)):
        survivor_names.append(survivors[s][0])
        
    
    return survivor_names
 
def single_cell_shuffle(bugs,survivor_names,gene_bag):
    #take bug0 and capture all of their cells for layer 1. And then put them in a dictionary for layers
    #bugn cells in layer 1 go to layer_1_cells dictionary.
    #bugn cells in layer 2 go to layer_2_cells dictionary.
    
    gene_bag["layer_1_cells_W1"] = []
    gene_bag["layer_1_cells_b1"] = []
    gene_bag["layer_2_cells_W2"] = []
    gene_bag["layer_2_cells_b2"] = []
    gene_bag["layer_3_cells_W3"] = []
    gene_bag["layer_3_cells_b3"] = []
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["W1"])):
            gene_bag["layer_1_cells_W1"].append(bugs[survivor_names[i]]["W1"][s,:])
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["b1"])):
            gene_bag["layer_1_cells_b1"].append(bugs[survivor_names[i]]["b1"][s,:])
            
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["W2"])):
            gene_bag["layer_2_cells_W2"].append(bugs[survivor_names[i]]["W2"][s,:])
    
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["b2"])):
            gene_bag["layer_2_cells_b2"].append(bugs[survivor_names[i]]["b2"][s,:])
    
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["decider_W3"])):
            gene_bag["layer_3_cells_W3"].append(bugs[survivor_names[i]]["decider_W3"][s,:])
    
    for i in range(len(survivor_names)):
        for s in range(len(bugs["bug0"]["decider_b3"])):
            gene_bag["layer_3_cells_b3"].append(bugs[survivor_names[i]]["decider_b3"][s,:])
    
    
    return gene_bag
 
def two_cell_shuffle(bugs,survivor_names,gene_bag):
    
    gene_bag["layer_1_2_cells_W1"] = []
    gene_bag["layer_1_2_cells_b1"] = []
    gene_bag["layer_2_2_cells_W2"] = []
    gene_bag["layer_2_2_cells_b2"] = []
    gene_bag["layer_3_2_cells_W3"] = []
    gene_bag["layer_3_2_cells_b3"] = []
    
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["W1"])),2):
            gene_bag["layer_1_2_cells_W1"].append(bugs[survivor_names[i]]["W1"][s:s+2,:])
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["b1"])),2):
            gene_bag["layer_1_2_cells_b1"].append(bugs[survivor_names[i]]["b1"][s:s+2,:])
            
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["W2"])),2):
            gene_bag["layer_2_2_cells_W2"].append(bugs[survivor_names[i]]["W2"][s:s+2,:])
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["b2"])),2):
            gene_bag["layer_2_2_cells_b2"].append(bugs[survivor_names[i]]["b2"][s:s+2,:])
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["decider_W3"])),2):
            gene_bag["layer_3_2_cells_W3"].append(bugs[survivor_names[i]]["decider_W3"][s:s+2,:])
    
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["decider_b3"])),2):
            gene_bag["layer_3_2_cells_b3"].append(bugs[survivor_names[i]]["decider_b3"][s:s+2,:])
            
            
    return gene_bag
        
    
def four_cell_shuffle(bugs,survivor_names,gene_bag):
    gene_bag["layer_1_4_cells_W1"] = []
    gene_bag["layer_1_4_cells_b1"] = []
    gene_bag["layer_2_4_cells_W2"] = []
    gene_bag["layer_2_4_cells_b2"] = []
    gene_bag["layer_3_4_cells_W3"] = []
    gene_bag["layer_3_4_cells_b3"] = []
    
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["W1"])),4):
            gene_bag["layer_1_4_cells_W1"].append(bugs[survivor_names[i]]["W1"][s:s+4,:])
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["b1"])),4):
            gene_bag["layer_1_4_cells_b1"].append(bugs[survivor_names[i]]["b1"][s:s+4,:])
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["W2"])),4):
            gene_bag["layer_2_4_cells_W2"].append(bugs[survivor_names[i]]["W2"][s:s+4,:])
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["b2"])),4):
            gene_bag["layer_2_4_cells_b2"].append(bugs[survivor_names[i]]["b2"][s:s+4,:])
            
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["decider_W3"])),4):
            gene_bag["layer_3_4_cells_W3"].append(bugs[survivor_names[i]]["decider_W3"][s:s+4,:])
            
            
    for i in range(len(survivor_names)):
        for s in range(0,int(len(bugs["bug0"]["decider_b3"])),4):
            gene_bag["layer_3_4_cells_b3"].append(bugs[survivor_names[i]]["decider_b3"][s:s+4,:])
            
            
    
    return gene_bag
 
def the_big_shuffler(bugs,survivor_names):
    gene_bag = {}
    
    gene_bag = single_cell_shuffle(bugs,survivor_names,gene_bag)
    
    gene_bag = two_cell_shuffle(bugs,survivor_names,gene_bag)
    
    gene_bag = four_cell_shuffle(bugs,survivor_names,gene_bag)
    
    return gene_bag
 
 
def single_cell_create(bugs,gene_bag):
    for i in range(0,len(bugs)//2):
        gene_W1 = []
        gene_b1 = []
        gene_W2 = []
        gene_b2 = []
        gene_W3 = []
        gene_b3 = []
        for k in range(len(bugs["bug0"]["W1"])):
            a = random.randint(0,len(gene_bag["layer_1_cells_W1"])-1)
            gene_W1.append(gene_bag["layer_1_cells_W1"][a])
             
            b = random.randint(0,len(gene_bag["layer_1_cells_b1"])-1)
            gene_b1.append(gene_bag["layer_1_cells_b1"][b])                
            
        for s in range(len(bugs["bug0"]["W2"])):
            c = random.randint(0,len(gene_bag["layer_2_cells_W2"])-1)
            gene_W2.append(gene_bag["layer_2_cells_W2"][c])
            
            d = random.randint(0,len(gene_bag["layer_2_cells_b2"])-1)
            gene_b2.append(gene_bag["layer_2_cells_b2"][d])
            
        for f in range(len(bugs["bug0"]["decider_W3"])):
            e = random.randint(0,len(gene_bag["layer_3_cells_W3"])-1)
            gene_W3.append(gene_bag["layer_3_cells_W3"][e])
            
            f = random.randint(0,len(gene_bag["layer_3_cells_b3"])-1)
            gene_b3.append(gene_bag["layer_3_cells_b3"][f])
                
        
        bugs["bug" + str(i)]["W1"] = np.vstack(gene_W1)
        bugs["bug" + str(i)]["b1"] = np.vstack(gene_b1)
        bugs["bug" + str(i)]["W2"] = np.vstack(gene_W2)
        bugs["bug" + str(i)]["b2"] = np.vstack(gene_b2)
        bugs["bug" + str(i)]["decider_W3"] = np.vstack(gene_W3)
        bugs["bug" + str(i)]["decider_b3"] = np.vstack(gene_b3)
        bugs["bug" + str(i)]["coordinates"] = np.ones((4,1))    
    return bugs
 
 
 
def  two_cell_create(bugs,gene_bag):
    for i in range(len(bugs) // 2,len(bugs) // 2 + (len(bugs) // 2 ) // 2):
        gene_W1 = []
        gene_b1 = []
        gene_W2 = []
        gene_b2 = []
        gene_W3 = []
        gene_b3 = []
        
        for k in range(len(bugs["bug0"]["W1"]) // 2):
            a = random.randint(0,len(gene_bag["layer_1_2_cells_W1"])-1)
            gene_W1.append(gene_bag["layer_1_2_cells_W1"][a])
                
            b = random.randint(0,len(gene_bag["layer_1_2_cells_b1"])-1)
            gene_b1.append(gene_bag["layer_1_2_cells_b1"][b])
            
        for s in range(len(bugs["bug0"]["W2"]) // 2):
        
            c = random.randint(0,len(gene_bag["layer_2_2_cells_W2"])-1)
            gene_W2.append(gene_bag["layer_2_2_cells_W2"][c])
            
            d = random.randint(0,len(gene_bag["layer_2_2_cells_b2"])-1)
            gene_b2.append(gene_bag["layer_2_2_cells_b2"][d])
            
        for j in range(0,len(bugs["bug0"]["decider_W3"]) // 2):
            e = random.randint(0,len(gene_bag["layer_3_2_cells_W3"])-1)
            gene_W3.append(gene_bag["layer_3_2_cells_W3"][e])
            
            f = random.randint(0,len(gene_bag["layer_3_2_cells_b3"])-1)
            gene_b3.append(gene_bag["layer_3_2_cells_b3"][f])
    
    
    
        bugs["bug" + str(i)]["W1"] = np.vstack(gene_W1)
        bugs["bug" + str(i)]["b1"] = np.vstack(gene_b1)
        bugs["bug" + str(i)]["W2"] = np.vstack(gene_W2)
        bugs["bug" + str(i)]["b2"] = np.vstack(gene_b2)
        #bugs["bug" + str(i)]["decider_W3"] = np.vstack(gene_W3)
        #bugs["bug" + str(i)]["decider_b3"] = np.vstack(gene_b3)
        bugs["bug" + str(i)]["coordinates"] = np.ones((4,1))
        
    return bugs
    
    
    
def four_cell_create(bugs,gene_bag):
        for i in range(len(bugs) // 2 + (len(bugs) // 2 ) // 2 , len(bugs) // 2 + (len(bugs) // 2 ) // 2 + (len(bugs) // 2) // 2):
            gene_W1 = []
            gene_b1 = []
            gene_W2 = []
            gene_b2 = []
            gene_W3 = []
            gene_b3 = []
            
            
            for k in range(len(bugs["bug0"]["W1"]) // 4):
                a = random.randint(0,len(gene_bag["layer_1_4_cells_W1"])-1)
                gene_W1.append(gene_bag["layer_1_4_cells_W1"][a])
                
                b = random.randint(0,len(gene_bag["layer_1_4_cells_b1"])-1)
                gene_b1.append(gene_bag["layer_1_4_cells_b1"][b])
            
            for s in range(len(bugs["bug0"]["W2"]) // 4):
        
                c = random.randint(0,len(gene_bag["layer_2_4_cells_W2"])-1)
                gene_W2.append(gene_bag["layer_2_4_cells_W2"][c])
            
                d = random.randint(0,len(gene_bag["layer_2_4_cells_b2"])-1)
                gene_b2.append(gene_bag["layer_2_4_cells_b2"][d])
            
            for j in range(0,len(bugs["bug0"]["decider_W3"]) // 4):
                e = random.randint(0,len(gene_bag["layer_3_4_cells_W3"])-1)
                gene_W3.append(gene_bag["layer_3_4_cells_W3"][e])
            
                f = random.randint(0,len(gene_bag["layer_3_4_cells_b3"])-1)
                gene_b3.append(gene_bag["layer_3_4_cells_b3"][f])
            
            
            
            bugs["bug" + str(i)]["W1"] = np.vstack(gene_W1)
            bugs["bug" + str(i)]["b1"] = np.vstack(gene_b1)
            bugs["bug" + str(i)]["W2"] = np.vstack(gene_W2)
            bugs["bug" + str(i)]["b2"] = np.vstack(gene_b2)
            #bugs["bug" + str(i)]["decider_W3"] = np.vstack(gene_W3)
            #bugs["bug" + str(i)]["decider_b3"] = np.vstack(gene_b3)
            bugs["bug" + str(i)]["coordinates"] = np.ones((4,1))
            
        return bugs
        

def move_turtle_wpoints(turtle_name,movement_point,turtles):
    
    turtles[turtle_name].setheading(90)
    turtles[turtle_name].forward(int(movement_point[0]))
    
    turtles[turtle_name].setheading(0)
    turtles[turtle_name].forward(int(movement_point[1]))
    
    turtles[turtle_name].setheading(180)
    turtles[turtle_name].forward(int(movement_point[2]))
    
    turtles[turtle_name].setheading(270)
    turtles[turtle_name].forward(int(movement_point[3]))
    
def move_all_turtles_wpoints(turtles,movement_points):
    for i in range(len(turtles)):
        move_turtle_wpoints("bug" + str(i),movement_points["bug" + str(i)],turtles)
def reset_the_turtles(turtles):
    
    for k in range(len(turtles)):
        turtles["bug" + str(k)].setposition(1,1)
        #turtles["bug" + str(k)].clear()
        #turtles["bug" + str(k)].reset()
        
    

def start_show(bugs,food,iteration_number):
    for i in range(0,iteration_number + 1):
        
        movement_points = all_the_bugs_deciding(bugs,food)
        #print(movement_points)
 
        
        #print("beginning: " ,bugs["bug0"]["coordinates"])
        move_all_the_bugs(bugs,movement_points)
        
        #print("last: " ,bugs["bug0"]["coordinates"])
        #print("movement:" ,movement_points["bug0"])
 
        evaluating_bugs(bugs,food)
 
        survivor_names = natural_selection(bugs,food)
        print(survivor_names)
        print(bugs["bug0"]["score"])
       
        if i < iteration_number:
            gene_bag = the_big_shuffler(bugs,survivor_names)
        #print(gene_bag)
 
            bugs = single_cell_create(bugs,gene_bag)
 
            bugs = two_cell_create(bugs,gene_bag)
 
            bugs = four_cell_create(bugs,gene_bag)
            
    return bugs,survivor_names




def start_show_visual(bugs,food,iteration_number,turtles):
    for i in range(0,iteration_number + 1):
        
        movement_points = all_the_bugs_deciding(bugs,food)
        #print(movement_points)
 
        move_all_turtles_wpoints(turtles,movement_points)
        
        if i < iteration_number:
            reset_the_turtles(turtles)
        
        #print("beginning: " ,bugs["bug0"]["coordinates"])
        move_all_the_bugs(bugs,movement_points)
        #print("last: " ,bugs["bug0"]["coordinates"])
        #print("movement:" ,movement_points["bug0"])
 
        evaluating_bugs(bugs,food)
 
        survivor_names = natural_selection(bugs,food)
        print(survivor_names)
        print(bugs["bug0"]["score"])
       
        if i < iteration_number:
            gene_bag = the_big_shuffler(bugs,survivor_names)
        #print(gene_bag)
 
            bugs = single_cell_create(bugs,gene_bag)
 
            bugs = two_cell_create(bugs,gene_bag)
 
            bugs = four_cell_create(bugs,gene_bag)
 
    return bugs,survivor_names
    
    
    
    
mode_name = input("would you like a visual or non-visual display")
#For convinience use numbers that could be divided by 4 as parameters.
if mode_name == "visual":
    food,food_turt = make_food_visual(6) #parameter is the number of food to be created
    data_length =food["fo0"].shape[0] + 4 #(This 4 symbolizes the four direction of movement)
    num_data = len(food)
     
    bugs,turtles = make_bugs_visual(1,100,8,data_length,num_data) # make_bugs(random_seed, number_of_bugs ,number_of_neurons_in_ the_beginning_layer )
    
    
    
    #print(bugs["bug1"])
    bugs,survivor_names = start_show_visual(bugs,food,100,turtles) # start_show(bugs,food, number_of_iterations(cycles))
    #print(bugs["bug0"])
    print(survivor_names)

    for i in range(len(bugs)):
        print(bugs["bug"+str(i)]["score"])
    
    turtle.mainloop()


elif mode_name == "non-visual":
    food = make_food(1) #parameter is the number of food to be created
    data_length =food["fo0"].shape[0] + 4 #(This 4 symbolizes the four direction of movement)
    num_data = len(food)
     
    bugs = make_bugs(1,20,8,data_length,num_data) # make_bugs(random_seed, number_of_bugs ,number_of_neurons_in_ the_beginning_layer )
    #print(bugs["bug1"])
    bugs,survivor_names = start_show(bugs,food,50) # start_show(bugs,food, number_of_iterations(cycles))
    #print(bugs["bug0"])
    print(survivor_names)

    for i in range(len(bugs)):
        print(bugs["bug"+str(i)]["score"])    
        
    