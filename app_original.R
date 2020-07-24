
ui = fluidPage(

    # Application title
    titlePanel("Japanese Handwritten Character Recognition"),
    h5("Kyle Erf"),
    h5("DS 501 - Summer 2020 - WPI"),
    h5("Final Project"),
    
    h3("Description"),
    
    h4("What data was used?"),
    h5("This Shiny App uses the ", a(href = "http://etlcdb.db.aist.go.jp/", "ETL Character Database", .noWS = "outside"), ", a collection of
       more than 1 million hand-written and machine-printed numerals, symbols, Latin alphabets, 
       and Japanese characters.", .noWS = c("after-begin", "before-end")),
    h5("In this interactive app, a dataset of over 60,000 images of Japanese katakana characters was used. This is a subset of the ETL-1 dataset.
       The quantity was reduced to 5500 images in order to meet Shiny App memory requirements."),
    
    h4("What was the motivation for choosing this topic?"),
    h5("I chose this dataset and project because I am interested in handwriting categorization through neural networks and I speak Japanese. 
       I have seen many examples of handwritten character recognition with numerals and English characters, so I was curious how well the
       same approach could be applied to Japanese characters. There are many real life applications for this type of data product; machine
       learning recognition of handwritten characters allows for quickly converting between manually collected data and digital data that
       can be manipulated more easily."),
    
    h4("How was the data analyzed?"),
    h5("The method of data analysis for this project is to create a neural net with Keras and Tensorflow that can recognize and 
       categorize image of Japanese characters, or attributes about the writer of the characters. In order to make this an interactive web app, many of the parameters for the model are all 
       available as editable user inputs."),
    h5("You can choose between 3 different variables to fit for with the \"Variable to Predict\" dropdown menu. 
       The three options are the character itself, the age of the participant, and the gender of the participant."),
    h5("You can also choose between 3 different options for Keras neural network models to compare accuracies.
       Each tab has a different model and different parameters you can adjust. The three options are a baseline 2-layer model, 
       a simple CNN model with one 2D Convolution layer, and a larger CNN model with two 2D Convolution layers."),
    h5("Click Run on a given tab to use the current settings; please be patient, the model fitting can be time-consuming 
       depending on the settings. The first time it is ran will take the longest to set up all the libraries, but subsequent tests
       should be shorter. To reduce time, consider decreasing the number of epochs or the size of the CNN filters."),
    
    h4("What did you find in the data?"),
    h5("Analysis of Raw Data: Most of the data came from male participants. Most of the participants came from the age range 20 to 30 years old. These skewed demographics
       come into play in the section below."),
    h5("Analysis with Neural Network: For characters, The baseline model worked fairly well with relatively few epochs, which typical test set accuracies of almost 90%.
       As expected, the Simple CNN model performed even better, with a typical test set accuracies closer to 94%. The Large CNN model had similar results to the Simple
       CNN model, indicating that the point of diminishing returns may have been reached, but I suspect that further optimizations to the model parameters would produce
       even better results than the Simple CNN model. One thing to note is that the number of characters to categorize (randomly chosen from the 46 or so total katakana characters)
       was limited to 4 for simplicity, but more could have been added."),
    h5("For age and gender, the results were much poorer. Gender had high accuracies for male participants but low accuracies for female participants.
       I believe that this is a result of the skewed populations of male and female participants, which you can see below. With more men overall, it
       is a safer guess for the model to predict the gender as male most of the time. A similar phenomenon occurs with the ages of participants.
       Because the population is so skewed towards the younger side, I had to choose an age of 25 years old as the cutoff for guesses; however, the
       model still predicted \"Up to age 25\" very frequently; this may have something to do with the keras model favoring a guess of \"0\" when it's
       not sure."),
    h5("I was particularly curious about the possibility of the model discerning differences in age and gender from the handwriting. However, I would summarize
       my findings with the conclusion that age and gender are not strongly discernable within this dataset. More data, especially more balanced data amongst the demographics,
       would go a long way towards making accuracte classification a reality. That being said, the neural networks were very successful in categorizing randomly
       selected katakana characters with all of the model types. This proves that the same approach applied to numerals and English characters can be applied to
       Japanese katakana characters as well."),
    hr(),
    
    h3("10 random input images of handwritten characters"),
    h5("This is an example of 10 random input images of handwritten katakana characters from the ETL-1 database, along with their English readings."),
    fluidRow(
        column(1, imageOutput("image1", width = 100, height = 100)), column(1, imageOutput("image2", width = 100, height = 100)),
        column(1, imageOutput("image3", width = 100, height = 100)), column(1, imageOutput("image4", width = 100, height = 100)),
        column(1, imageOutput("image5", width = 100, height = 100)), column(1, imageOutput("image6", width = 100, height = 100)),
        column(1, imageOutput("image7", width = 100, height = 100)), column(1, imageOutput("image8", width = 100, height = 100)),
        column(1, imageOutput("image9", width = 100, height = 100)), column(3, imageOutput("image10", width = 100, height = 100))
    ),
    fluidRow(
        column(1, textOutput("char1"), align = "center"), column(1, textOutput("char2"), align = "center"),
        column(1, textOutput("char3"), align = "center"), column(1, textOutput("char4"), align = "center"),
        column(1, textOutput("char5"), align = "center"), column(1, textOutput("char6"), align = "center"),
        column(1, textOutput("char7"), align = "center"), column(1, textOutput("char8"), align = "center"),
        column(1, textOutput("char9"), align = "center"), column(1, textOutput("char10"), align = "center")
    ),
    # sample_images[1],
    hr(),
    
    h3("Analysis of All Input Data"),
    h5("These are distributions showing the demographics of the writers of the data in the images - age and gender."),
    fluidRow(
        column(6, plotOutput("ageDistPlot")),
        column(6, plotOutput("genderDistPlot"))
    ),
    hr(),
    
    h3("Model Building and Visualization"),
    selectInput("select_fit", "Variable to Predict:", c("Characters" = "Characters", "Age" = "Age", "Gender" = "Gender")),
    tabsetPanel(type = "tabs", id = "tabs",
        tabPanel("Baseline Model",
             sidebarLayout(
                 sidebarPanel(
                     # numericInput("n_chars_baseline", "Number of Characters to Recognize:", 4, min = 2, max = 4),
                     numericInput("n_epochs_baseline", "Number of Epochs:", 5, min = 1, max = 30),
                     numericInput("batch_size_baseline", "Batch size:", 200, min = 10, max = 500),

                     actionButton("baseline_run", "Run")
                 ),
                 mainPanel(
                     h4("Training Set:"),
                     imageOutput("baseline_modelPlot"),
                     h4("Test Set:"),
                     h5("Table of test set instances and frequencies:"),
                     tableOutput("baseline_table"),
                     textOutput("baseline_accuracy"),
                     # textOutput("choices_baseline")
                 )
             )
        ),
        tabPanel("Simple CNN Model",
            sidebarLayout(
                sidebarPanel(
                    # numericInput("n_chars_simple", "Number of Characters to Recognize:", 4, min = 2, max = 4),
                    numericInput("n_epochs_simple", "Number of Epochs:", 10, min = 5, max = 100),
                    numericInput("batch_size_simple", "Batch size:", 200, min = 10, max = 500),

                    numericInput("l1_filters_simple", "2D Conv Layer 1 Output Filters:", 32, min = 10, max = 100),
                    numericInput("l1_kernel_simple", "2D Conv Layer 1 Kernel Size:", 5, min = 1, max = 10),
                    numericInput("l1_dropout_simple", "2D Conv Layer 1 Dropout Rate:", 0.2, min = 0, max = 0.9),

                    numericInput("l2_units_simple", "Dense Layer 2 Units:", 128, min = 1, max = 1000),

                    actionButton("simple_cnn_run", "Run")
                ),
                mainPanel(
                    h4("Training Set:"),
                    imageOutput("simple_modelPlot"),
                    h4("Test Set:"),
                    h5("Table of test set instances and frequencies:"),
                    tableOutput("simple_table"),
                    textOutput("simple_accuracy"),
                    # textOutput("choices_simple")
                )
            )
        ),
        tabPanel("Large CNN Model",
             sidebarLayout(
                 sidebarPanel(
                     # numericInput("n_chars_large", "Number of Characters to Recognize:", 4, min = 2, max = 4),
                     numericInput("n_epochs_large", "Number of Epochs:", 10, min = 5, max = 100),
                     numericInput("batch_size_large", "Batch size:", 200, min = 10, max = 500),

                     numericInput("l1_units_large", "2D Conv Layer 1 Output Filters:", 30, min = 10, max = 100),
                     numericInput("l1_kernel_large", "2D Conv Layer 1 Kernel Size:", 5, min = 1, max = 10),
                     numericInput("l1_pool_large", "2D Conv Layer 1 Pool Size:", 2, min = 1, max = 10),

                     numericInput("l2_units_large", "2D Conv Layer 2 Output Filters:", 15, min = 10, max = 100),
                     numericInput("l2_kernel_large", "2D Conv Layer 2 Kernel Size:", 3, min = 1, max = 10),
                     numericInput("l2_pool_large", "2D Conv Layer 2 Pool Size:", 2, min = 1, max = 10),
                     numericInput("l2_dropout_large", "2D Conv Layer 2 Dropout Rate:", 0.2, min = 0, max = 0.9),

                     numericInput("l3_units_large", "Dense Layer 3 Units:", 128, min = 1, max = 1000),

                     numericInput("l4_units_large", "Dense Layer 4 Units:", 50, min = 1, max = 1000),

                     actionButton("large_cnn_run", "Run")
                 ),
                 mainPanel(
                     h4("Training Set:"),
                     imageOutput("large_modelPlot"),
                     h4("Test Set:"),
                     h5("Table of test set instances and frequencies:"),
                     tableOutput("large_table"),
                     textOutput("large_accuracy"),
                     # textOutput("choices_large")
                 )
             )
        )
    ),
    hr(),
    
    h3("Percentage Correct per Option"),
    h5("Once a model has been run above, the plot below will show the rate at which each guess was correctly categorized in the test set."),
    imageOutput("wrong_guesses_plot"),
    # h5("Once a model has been run above, the plot below will show the guesses versus their correct answer, with the size of the dot representing the quantity of guesses at that point."),
    # imageOutput("predictions_plot")
)

# Define server logic required to draw a histogram
server = function(input, output) {
    library(stringr)
    library(shiny)
    library(shinyjs)
    library(dplyr)
    library(keras)
    library(caret)
    library(ggplot2)
    library(hash)
    library(lobstr)
    
    # set.seed(123)
    
    # Display Sample Images and Character Names
    image_folder = "sample_images/"
    images = list.files(image_folder)
    sample_images = sample(images, 10)
    sample_characters = c()
    for (i in 1:length(sample_images)) {
        new_character = str_split(sample_images[i], "_")[[1]][1]
        sample_characters = c(sample_characters, str_split(sample_images[i], "_")[[1]][1])
    }
    output$char1 <- renderText({sample_characters[1]})
    output$char2 <- renderText({sample_characters[2]})
    output$char3 <- renderText({sample_characters[3]})
    output$char4 <- renderText({sample_characters[4]})
    output$char5 <- renderText({sample_characters[5]})
    output$char6 <- renderText({sample_characters[6]})
    output$char7 <- renderText({sample_characters[7]})
    output$char8 <- renderText({sample_characters[8]})
    output$char9 <- renderText({sample_characters[9]})
    output$char10 <- renderText({sample_characters[10]})
    
    # Main Function
    
    if (!exists("ETL_data")) {
        ETL_data = readRDS("ETL_data.rds")
    }
    
    image_width = 64
    image_height = 63
    image_size = image_width * image_height
    
    print("Finished loading")
    
    observeEvent(input$baseline_run,{
        progress_baseline = shiny::Progress$new()
        on.exit(progress_baseline$close())
        
        progress_baseline$set(message = "Please Wait. Fitting Baseline Model...")
        
        # User Input
        fit_variable = input$select_fit
        if(fit_variable == "Characters") {
            n_codes = 4
        } else if(fit_variable == "Age") {
            n_codes = 2
        } else if(fit_variable == "Gender") {
            n_codes = 2
        }

        # Subset Data based on User Input
        char_subset_baseline = sample(unique(ETL_data$character), n_codes)
        
        all_chars_baseline = paste(char_subset_baseline, collapse = ", ")
        if(fit_variable == "Characters") {
            output$choices_baseline = renderText({paste("Characters to Recognize: ", all_chars_baseline, sep = "")})
        } else if(fit_variable == "Age") {
            output$choices_baseline = renderText({""})
        } else if(fit_variable == "Gender") {
            output$choices_baseline = renderText({""})
        }
        
        ETL_data_subset = data.frame()
        for (i in 1:n_codes) {
            next_char_data = ETL_data[ETL_data$character == char_subset_baseline[i],]
            ETL_data_subset = rbind(ETL_data_subset, next_char_data)
        }
        
        if(fit_variable == "Characters") {
            char_to_code = hash()
            counter = 0
            for (char in unique(ETL_data_subset$character)) {
                char_to_code[[char]] = counter
                counter = counter + 1
            }
            codes = c()
            for (char in ETL_data_subset$character) {
                codes = c(codes, char_to_code[[char]])
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Age") {
            codes = c()
            for (age in ETL_data_subset$age) {
                if(age <= 25) {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Gender") {
            codes = c()
            for (gender in ETL_data_subset$gender) {
                if(gender == "M") {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
        }
        
        # Extract input and category data
        trainIndex = createDataPartition(ETL_data_subset$code, p = 0.8, list=FALSE)
        training = ETL_data_subset[trainIndex,] # training input data (80% of data)
        testing = ETL_data_subset[-trainIndex,] # testing input data (20% of data)
        
        print("Pre-Clean")
        print(mem_used())
        
        # Reshape Input data
        x_train = training$image
        x_test = testing$image
        x_train = array_reshape(x_train, c(length(x_train), image_size))
        x_test = array_reshape(x_test, c(length(x_test), image_size))
        
        print("Mid-Clean")
        print(mem_used())
        
        # One-hot encode categories
        y_train = training$code
        y_test = testing$code
        y_train_onehot = to_categorical(y_train, n_codes)
        y_test_onehot = to_categorical(y_test, n_codes)
        
        print("Pre-Model")
        
        # Baseline Model
        baseline_model = keras_model_sequential()
        baseline_model %>%
            layer_dense(units = image_size, activation = "relu", input_shape = c(image_size)) %>%
            layer_dense(units = n_codes, activation = "softmax")

        baseline_model %>% compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizer_adam(),
            metrics = c('accuracy')
        )

        history_baseline = baseline_model %>% fit(
            x_train, y_train_onehot,
            epochs = input$n_epochs_baseline, batch_size = input$batch_size_baseline,
            validation_split = 0.2,
            verbose = 2
        )
        
        print("Post-Fit")

        test_accuracy_baseline = baseline_model %>% evaluate(x_test, y_test_onehot, verbose = 0)
        
        output$baseline_modelPlot = renderPlot({
            plot(history_baseline)
        })
        
        output$baseline_accuracy = renderText({
            paste("Baseline Model Accuracy (Test Set): ", round(test_accuracy_baseline[2] * 100, 2), "%")
        })
        
        model_guesses = baseline_model %>% predict_classes(x_test)
        wrong_indices = which(model_guesses != y_test)
        
        if(fit_variable == "Characters") {
            wrong_guesses = data.frame(testing$character[wrong_indices])
            wrong_guesses_table = table(wrong_guesses)
            if(length(names(wrong_guesses_table)) != length(char_subset_baseline)) {
                missing = setdiff(char_subset_baseline, names(wrong_guesses_table))
                wrong_guesses_table[missing] = 0
            }
            accuracy_table = data.frame(cbind(wrong_guesses_table, table(testing$character)))
            
        } else if(fit_variable == "Age") {
            wrong_guesses = data.frame(testing$age[wrong_indices])
            
            count0 = 0
            count1 = 0
            freqs = table(wrong_guesses)
            row_names = as.numeric(row.names(table(wrong_guesses)))
            for(i in 1:length(row_names)) {
                if(row_names[i] <= 25) {
                    count0 = count0 + freqs[[i]]
                } else {
                    count1 = count1 + freqs[[i]]
                }
            }
            
            count0d = 0
            count1d = 0
            freqs_d = table(testing$age)
            row_names_d = as.numeric(row.names(table(testing$age)))
            for(i in 1:length(row_names_d)) {
                if(row_names_d[i] <= 25) {
                    count0d = count0d + freqs_d[[i]]
                } else {
                    count1d = count1d + freqs_d[[i]]
                }
            }
            accuracy_table = data.frame(cbind(c(count0, count1), c(count0d, count1d)))
            row.names(accuracy_table) = c("Up to age 25", "Above age 25")
            
        } else if(fit_variable == "Gender") {
            wrong_guesses = data.frame(testing$gender[wrong_indices])
            accuracy_table = data.frame(cbind(table(wrong_guesses), table(testing$gender)))
            
        }
        
        names(accuracy_table) <- c("number_right", "total_guesses")
        accuracy_table$number_right = accuracy_table$total_guesses - accuracy_table$number_right
        accuracy_table = transform(accuracy_table, Percentage = number_right / total_guesses * 100)
        fit_var = row.names(accuracy_table)
        accuracy_table = cbind(row.names(accuracy_table), accuracy_table)
        names(accuracy_table)[names(accuracy_table) == "row.names(accuracy_table)"] = "Instance"
        
        output$baseline_table = renderTable({accuracy_table})
        
        output$wrong_guesses_plot = renderPlot({
            ggplot(accuracy_table, aes(x = fit_var, y = Percentage, fill = fit_var)) + geom_bar(stat = "identity", color = "black") + scale_fill_brewer((palette = "Blues"))
        })
        
        predictions = data.frame(cbind(y_test, model_guesses))
        predictions_freq = predictions %>% count(y_test, model_guesses)
        
        output$predictions_plot = renderPlot({
            ggplot(predictions_freq, aes(x = y_test, y = model_guesses)) + geom_point(aes(size = n, color = y_test))
        })
    })
    
    observeEvent(input$simple_cnn_run,{
        progress_simple = shiny::Progress$new()
        on.exit(progress_simple$close())
        
        progress_simple$set(message = "Please Wait. Fitting Simple CNN Model...")
        
        # User Input
        # n_codes = input$n_chars_simple
        fit_variable = input$select_fit
        if(fit_variable == "Characters") {
            n_codes = 4
        } else if(fit_variable == "Age") {
            n_codes = 2
        } else if(fit_variable == "Gender") {
            n_codes = 2
        }
        
        # Subset Data based on User Input
        char_subset_simple = sample(unique(ETL_data$character), n_codes)
        
        all_chars_simple = paste(char_subset_simple, collapse = ", ")
        if(fit_variable == "Characters") {
            output$choices_simple = renderText({paste("Characters to Recognize: ", all_chars_simple, sep = "")})
        } else if(fit_variable == "Age") {
            output$choices_simple = renderText({""})
        } else if(fit_variable == "Gender") {
            output$choices_simple = renderText({""})
        }
        
        ETL_data_subset = data.frame()
        for (i in 1:n_codes) {
            next_char_data = ETL_data[ETL_data$character == char_subset_simple[i],]
            ETL_data_subset = rbind(ETL_data_subset, next_char_data)
        }
        
        if(fit_variable == "Characters") {
            char_to_code = hash()
            counter = 0
            for (char in unique(ETL_data_subset$character)) {
                char_to_code[[char]] = counter
                counter = counter + 1
            }
            codes = c()
            for (char in ETL_data_subset$character) {
                codes = c(codes, char_to_code[[char]])
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Age") {
            codes = c()
            for (age in ETL_data_subset$age) {
                if(age <= 25) {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Gender") {
            codes = c()
            for (gender in ETL_data_subset$gender) {
                if(gender == "M") {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
        }
        
        # Extract input and category data
        trainIndex = createDataPartition(ETL_data_subset$code, p = 0.8, list=FALSE)
        training = ETL_data_subset[trainIndex,] # training input data (80% of data)
        testing = ETL_data_subset[-trainIndex,] # testing input data (20% of data)
        
        # Reshape Input data
        x_train = training$image
        x_test = testing$image
        x_train = array_reshape(x_train, c(length(x_train), image_width, image_height, 1))
        x_test = array_reshape(x_test, c(length(x_test), image_width, image_height, 1))
        
        # # One-hot encode categories
        y_train = training$code
        y_test = testing$code
        y_train_onehot = to_categorical(y_train, n_codes)
        y_test_onehot = to_categorical(y_test, n_codes)
        
        # Simple CNN
        simple_cnn = keras_model_sequential()
        simple_cnn %>%
            layer_conv_2d(input$l1_filters_simple, kernel_size = c(input$l1_kernel_simple, input$l1_kernel_simple), activation = "relu", input_shape = c(64, 63, 1)) %>%
            layer_max_pooling_2d() %>%
            layer_dropout(input$l1_dropout_simple) %>%
            layer_flatten() %>%
            layer_dense(input$l2_units_simple, activation = "relu") %>%
            layer_dense(n_codes, activation = "softmax")
        
        simple_cnn %>% compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizer_adam(),
            metrics = c('accuracy')
        )
        
        history_simple = simple_cnn %>% fit(
            x_train, y_train_onehot,
            epochs = input$n_epochs_simple, batch_size = input$batch_size_simple,
            validation_split = 0.2,
            verbose = 2
        )
        
        test_accuracy_simple = simple_cnn %>% evaluate(x_test, y_test_onehot, verbose = 0)
        
        output$simple_modelPlot = renderPlot({
            plot(history_simple)
        })
        
        test_freqs = data.frame(table(y_test))
        
        output$simple_accuracy = renderText({
            paste("Simple CNN Model Accuracy (Test Set): ", round(test_accuracy_simple[2] * 100, 2), "%")
        })
        
        model_guesses = simple_cnn %>% predict_classes(x_test)
        wrong_indices = which(model_guesses != y_test)
        
        if(fit_variable == "Characters") {
            wrong_guesses = data.frame(testing$character[wrong_indices])
            wrong_guesses_table = table(wrong_guesses)
            # if(length(names(wrong_guesses_table)) != length(char_subset_baseline)) {
            #     missing = setdiff(char_subset_baseline, names(wrong_guesses_table))
            #     wrong_guesses_table[missing] = 0
            # }
            accuracy_table = data.frame(cbind(wrong_guesses_table, table(testing$character)))
            
        } else if(fit_variable == "Age") {
            wrong_guesses = data.frame(testing$age[wrong_indices])
            
            count0 = 0
            count1 = 0
            freqs = table(wrong_guesses)
            row_names = as.numeric(row.names(table(wrong_guesses)))
            for(i in 1:length(row_names)) {
                if(row_names[i] <= 25) {
                    count0 = count0 + freqs[[i]]
                } else {
                    count1 = count1 + freqs[[i]]
                }
            }
            
            count0d = 0
            count1d = 0
            freqs_d = table(testing$age)
            row_names_d = as.numeric(row.names(table(testing$age)))
            for(i in 1:length(row_names_d)) {
                if(row_names_d[i] <= 25) {
                    count0d = count0d + freqs_d[[i]]
                } else {
                    count1d = count1d + freqs_d[[i]]
                }
            }
            accuracy_table = data.frame(cbind(c(count0, count1), c(count0d, count1d)))
            row.names(accuracy_table) = c("Up to age 25", "Above age 25")
            
        } else if(fit_variable == "Gender") {
            wrong_guesses = data.frame(testing$gender[wrong_indices])
            accuracy_table = data.frame(cbind(table(wrong_guesses), table(testing$gender)))
            
        }
        
        names(accuracy_table) <- c("number_right", "total_guesses")
        accuracy_table$number_right = accuracy_table$total_guesses - accuracy_table$number_right
        accuracy_table = transform(accuracy_table, Percentage = number_right / total_guesses * 100)
        fit_var = row.names(accuracy_table)
        accuracy_table = cbind(row.names(accuracy_table), accuracy_table)
        names(accuracy_table)[names(accuracy_table) == "row.names(accuracy_table)"] = "Instance"
        
        output$simple_table = renderTable({accuracy_table})
        
        output$wrong_guesses_plot = renderPlot({
            ggplot(accuracy_table, aes(x = fit_var, y = Percentage, fill = fit_var)) + geom_bar(stat = "identity", color = "black") + scale_fill_brewer((palette = "Blues"))
        })
        
        predictions = data.frame(cbind(y_test, model_guesses))
        predictions_freq = predictions %>% count(y_test, model_guesses)
        
        output$predictions_plot = renderPlot({
            ggplot(predictions_freq, aes(x = y_test, y = model_guesses)) + geom_point(aes(size = n, color = y_test))
        })
    })
    
    observeEvent(input$large_cnn_run,{
        progress_large = shiny::Progress$new()
        on.exit(progress_large$close())
        
        progress_large$set(message = "Please Wait. Fitting Large CNN Model...")
        
        # User Input
        # n_codes = input$n_chars_large
        fit_variable = input$select_fit
        if(fit_variable == "Characters") {
            n_codes = 4
        } else if(fit_variable == "Age") {
            n_codes = 2
        } else if(fit_variable == "Gender") {
            n_codes = 2
        }
        
        # Subset Data based on User Input
        char_subset_large = sample(unique(ETL_data$character), n_codes)
        
        all_chars_large = paste(char_subset_large, collapse = ", ")
        if(fit_variable == "Characters") {
            output$choices_large = renderText({paste("Characters to Recognize: ", all_chars_large, sep = "")})
        } else if(fit_variable == "Age") {
            output$choices_large = renderText({""})
        } else if(fit_variable == "Gender") {
            output$choices_large = renderText({""})
        }
        
        ETL_data_subset = data.frame()
        for (i in 1:n_codes) {
            next_char_data = ETL_data[ETL_data$character == char_subset_large[i],]
            ETL_data_subset = rbind(ETL_data_subset, next_char_data)
        }
        
        if(fit_variable == "Characters") {
            char_to_code = hash()
            counter = 0
            for (char in unique(ETL_data_subset$character)) {
                char_to_code[[char]] = counter
                counter = counter + 1
            }
            codes = c()
            for (char in ETL_data_subset$character) {
                codes = c(codes, char_to_code[[char]])
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Age") {
            codes = c()
            for (age in ETL_data_subset$age) {
                if(age <= 25) {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
            
        } else if(fit_variable == "Gender") {
            codes = c()
            for (gender in ETL_data_subset$gender) {
                if(gender == "M") {
                    codes = c(codes, 0)
                } else {
                    codes = c(codes, 1)
                }
            }
            ETL_data_subset$code = codes
        }
        
        # Extract input and category data
        trainIndex = createDataPartition(ETL_data_subset$code, p = 0.8, list=FALSE)
        training = ETL_data_subset[trainIndex,] # training input data (80% of data)
        testing = ETL_data_subset[-trainIndex,] # testing input data (20% of data)
        
        # Reshape Input data
        x_train = training$image
        x_test = testing$image
        x_train = array_reshape(x_train, c(length(x_train), image_width, image_height, 1))
        x_test = array_reshape(x_test, c(length(x_test), image_width, image_height, 1))
        
        # # One-hot encode categories
        y_train = training$code
        y_test = testing$code
        y_train_onehot = to_categorical(y_train, n_codes)
        y_test_onehot = to_categorical(y_test, n_codes)
        
        # Large CNN
        large_cnn = keras_model_sequential()
        large_cnn %>%
            layer_conv_2d(input$l1_units_large, kernel_size = c(input$l1_kernel_large, input$l1_kernel_large), activation = "relu", input_shape = c(64, 63, 1)) %>%
            layer_max_pooling_2d(pool_size = c(input$l1_pool_large, input$l1_pool_large)) %>%
            layer_conv_2d(input$l2_units_large, kernel_size = c(input$l2_kernel_large, input$l2_kernel_large), activation = "relu") %>%
            layer_max_pooling_2d(pool_size = c(input$l2_pool_large, input$l2_pool_large)) %>%
            layer_dropout(input$l2_dropout_large) %>%
            layer_flatten() %>%
            layer_dense(input$l3_units_large, activation = "relu") %>%
            layer_dense(input$l4_units_large, activation = "relu") %>%
            layer_dense(n_codes, activation = "softmax")
        
        large_cnn %>% compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizer_adam(),
            metrics = c('accuracy')
        )
        
        history_large = large_cnn %>% fit(
            x_train, y_train_onehot,
            epochs = input$n_epochs_large, batch_size = input$batch_size_large,
            validation_split = 0.2,
            verbose = 2
        )
        
        test_accuracy_large = large_cnn %>% evaluate(x_test, y_test_onehot, verbose = 0)
        
        output$large_modelPlot = renderPlot({
            plot(history_large)
        })
        
        test_freqs = data.frame(table(y_test))
        
        output$large_accuracy = renderText({
            paste("Large CNN Model Accuracy (Test Set): ", round(test_accuracy_large[2] * 100, 2), "%")
        })
        
        model_guesses = large_cnn %>% predict_classes(x_test)
        wrong_indices = which(model_guesses != y_test)
        
        if(fit_variable == "Characters") {
            wrong_guesses = data.frame(testing$character[wrong_indices])
            wrong_guesses_table = table(wrong_guesses)
            # if(length(names(wrong_guesses_table)) != length(char_subset_baseline)) {
            #     missing = setdiff(char_subset_baseline, names(wrong_guesses_table))
            #     wrong_guesses_table[missing] = 0
            # }
            accuracy_table = data.frame(cbind(wrong_guesses_table, table(testing$character)))
            
        } else if(fit_variable == "Age") {
            wrong_guesses = data.frame(testing$age[wrong_indices])
            
            count0 = 0
            count1 = 0
            freqs = table(wrong_guesses)
            row_names = as.numeric(row.names(table(wrong_guesses)))
            for(i in 1:length(row_names)) {
                if(row_names[i] <= 25) {
                    count0 = count0 + freqs[[i]]
                } else {
                    count1 = count1 + freqs[[i]]
                }
            }
            
            count0d = 0
            count1d = 0
            freqs_d = table(testing$age)
            row_names_d = as.numeric(row.names(table(testing$age)))
            for(i in 1:length(row_names_d)) {
                if(row_names_d[i] <= 25) {
                    count0d = count0d + freqs_d[[i]]
                } else {
                    count1d = count1d + freqs_d[[i]]
                }
            }
            accuracy_table = data.frame(cbind(c(count0, count1), c(count0d, count1d)))
            row.names(accuracy_table) = c("Up to age 25", "Above age 25")
            
        } else if(fit_variable == "Gender") {
            wrong_guesses = data.frame(testing$gender[wrong_indices])
            accuracy_table = data.frame(cbind(table(wrong_guesses), table(testing$gender)))
            
        }
        
        names(accuracy_table) <- c("number_right", "total_guesses")
        accuracy_table$number_right = accuracy_table$total_guesses - accuracy_table$number_right
        accuracy_table = transform(accuracy_table, Percentage = number_right / total_guesses * 100)
        fit_var = row.names(accuracy_table)
        accuracy_table = cbind(row.names(accuracy_table), accuracy_table)
        names(accuracy_table)[names(accuracy_table) == "row.names(accuracy_table)"] = "Instance"
        
        output$large_table = renderTable({accuracy_table})
        
        output$wrong_guesses_plot = renderPlot({
            ggplot(accuracy_table, aes(x = fit_var, y = Percentage, fill = fit_var)) + geom_bar(stat = "identity", color = "black") + scale_fill_brewer((palette = "Blues"))
        })
        
        predictions = data.frame(cbind(y_test, model_guesses))
        predictions_freq = predictions %>% count(y_test, model_guesses)
        
        output$predictions_plot = renderPlot({
            ggplot(predictions_freq, aes(x = y_test, y = model_guesses)) + geom_point(aes(size = n, color = y_test))
        })
    })
    
    output$ageDistPlot = renderPlot({
        ggplot(ETL_data, aes(x = age, fill = age)) + geom_histogram(binwidth = 1, color = "black", fill = "steelblue")
    })
    output$genderDistPlot = renderPlot({
        ggplot(ETL_data, aes(x = gender, fill = gender)) + geom_bar(color = "black") + scale_fill_brewer(palette = "Blues")
    })
    
    # Sample Images
    output$image1 = renderImage({
        return(list(src = paste(image_folder, sample_images[1], sep = "")))
    }, deleteFile = FALSE)
    output$image2 = renderImage({
        return(list(src = paste(image_folder, sample_images[2], sep = "")))
    }, deleteFile = FALSE)
    output$image3 = renderImage({
        return(list(src = paste(image_folder, sample_images[3], sep = "")))
    }, deleteFile = FALSE)
    output$image4 = renderImage({
        return(list(src = paste(image_folder, sample_images[4], sep = "")))
    }, deleteFile = FALSE)
    output$image5 = renderImage({
        return(list(src = paste(image_folder, sample_images[5], sep = "")))
    }, deleteFile = FALSE)
    output$image6 = renderImage({
        return(list(src = paste(image_folder, sample_images[6], sep = "")))
    }, deleteFile = FALSE)
    output$image7 = renderImage({
        return(list(src = paste(image_folder, sample_images[7], sep = "")))
    }, deleteFile = FALSE)
    output$image8 = renderImage({
        return(list(src = paste(image_folder, sample_images[8], sep = "")))
    }, deleteFile = FALSE)
    output$image9 = renderImage({
        return(list(src = paste(image_folder, sample_images[9], sep = "")))
    }, deleteFile = FALSE)
    output$image10 = renderImage({
        return(list(src = paste(image_folder, sample_images[10], sep = "")))
    }, deleteFile = FALSE)
    
    # Model Plots
    output$baseline_deepviz = renderImage({
        return(list(src = "C:/Users/kerf/OneDrive - Analog Devices, Inc/WPI/DS501/Final Project/baseline_model_deepviz.png"))
    })
    output$simple_deepviz = renderImage({
        return(list(src = "C:/Users/kerf/OneDrive - Analog Devices, Inc/WPI/DS501/Final Project/simple_model_deepviz.png"))
    })
    output$large_deepviz = renderImage({
        return(list(src = "C:/Users/kerf/OneDrive - Analog Devices, Inc/WPI/DS501/Final Project/large_model_deepviz.png"))
    })
    
}

# Run the application
shinyApp(ui = ui, server = server)
