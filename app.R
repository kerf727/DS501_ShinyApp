
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
       categorize image of Japanese characters, or attributes about the writer of the characters. The data has been pre-trained and fitted
       with 20 epochs each."),
    h5("You can choose between 3 different variables to fit for with the \"Variable to Predict\" dropdown menu. 
       The three options are the character itself, the age of the participant, and the gender of the participant."),
    h5("You can also choose between 3 different options for Keras neural network models to compare accuracies.
        The three options are a baseline 2-layer model, a simple CNN model with one 2D Convolution layer, and a larger CNN model with two 2D Convolution layers."),
    
    h4("What did the data show?"),
    h5("Analysis of Raw Data: Most of the data came from male participants. Most of the participants came from the age range 20 to 30 years old. These skewed demographics
       come into play in the section below."),
    h5("Analysis with Neural Network: For characters, The baseline model worked quite well with relatively few epochs, which typical test set accuracies of almost 90%.
       As expected, the Simple CNN model performed even better, with a typical test set accuracies closer to 97%. The Large CNN model had similar results (also about 97%) to the Simple
       CNN model, indicating that the point of diminishing returns may have been reached, but I suspect that further optimizations to the model parameters would produce
       even better results than the Simple CNN model. One thing to note is that the number of characters to categorize (randomly chosen from the 46 or so total katakana characters)
       was limited to 4 for simplicity, but more could have been added. These were randomized for every run, so thre three models were trained on different characters."),
    h5("For age and gender, the results were much poorer. Gender had high accuracies for male participants but low accuracies for female participants.
       I believe that this is a result of the skewed populations of male and female participants, which you can see below. With more men overall, it
       is a safer guess for the model to predict the gender as male most of the time. A similar phenomenon occurs with the ages of participants.
       Because the population is so skewed towards the younger side, I had to choose an age of 25 years old as the cutoff for guesses; however, the
       model still predicted \"Up to age 25\" very frequently; this may have something to do with the keras model favoring a guess of \"0\" when it's not confident."),
    h5("I was particularly curious about the possibility of the model discerning differences in age and gender from the handwriting. However, I would summarize
       my findings with the conclusion that age and gender are not strongly discernable within this dataset. The test set accuracies were about 50% in all of the age tests and about 70% in all of the gender tests,
       regardless of model used. This is nearly equivalent to guessing one option or the other. In addition, the validation accuracies were very low while the training accuracies were not bad,
       indicating that the models were overfitted; there is definitely room for improvement in the models themselves, but I think that more data, especially more balanced data amongst the demographics,
       would go a long way towards making accuracte classification a reality. That being said, the neural networks were very successful in categorizing randomly
       selected katakana characters with all of the model types. This proves that the same approach applied to numerals and English characters can be applied to
       Japanese katakana characters as well."),
    h5("For next steps for this project, given more time, I would add functionality to test categorizing more characters at once and also moving on to other Japanese characters,
       such as the more complicated kanji (Chinese origin characters). More work in balancing the demographics of the input data might also make categorizing age or gender more successful."),
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
    hr(),
    
    h3("Analysis of Input Data"),
    h5("These are distributions showing the demographics of the writers of the data in the images - age and gender."),
    fluidRow(
        imageOutput("age_gender")
    ),
    hr(),
    
    h3("Model Evaluation"),
    selectInput("select_fit", "Variable to Predict:", c("Characters" = "Characters", "Age" = "Age", "Gender" = "Gender")),
    tabsetPanel(type = "tabs", id = "tabs",
        tabPanel("Baseline Model",
            fluidRow(
                column(4, imageOutput("baseline_model")),
                column(6, imageOutput("baseline_plot")),
            )
        ),
        tabPanel("Simple CNN Model",
             fluidRow(
                 column(4, imageOutput("simple_model")),
                 column(6, imageOutput("simple_plot")),
             )
        ),
        tabPanel("Large CNN Model",
             fluidRow(
                 column(4, imageOutput("large_model")),
                 column(6, imageOutput("large_plot")),
             )
        )
    )
)

# Define server logic required to draw a histogram
server = function(input, output) {
    library(stringr)
    
    # Display Sample Images and Character Names
    image_folder = "sample_images/"
    images = list.files(image_folder)
    sample_images = sample(images, 10)
    sample_characters = c()
    for (i in 1:length(sample_images)) {
        new_character = str_split(sample_images[i], "_")[[1]][1]
        sample_characters = c(sample_characters, str_split(sample_images[i], "_")[[1]][1])
    }
    
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
    
    # Histograms
    output$age_gender = renderImage({return(list(src = "sample_images/age_gender_histograms.png"))}, deleteFile = FALSE)
    
    output$baseline_model = renderImage({return(list(src = "model_images/baseline_model.png"))}, deleteFile = FALSE)
    output$simple_model = renderImage({return(list(src = "model_images/simple_model.png"))}, deleteFile = FALSE)
    output$large_model = renderImage({return(list(src = "model_images/large_model.png"))}, deleteFile = FALSE)
    
    # Accuracy plots
    observeEvent(input$select_fit, {
        
        if (input$select_fit == "Characters") {
            output$baseline_plot = renderImage({return(list(src = "model_images/baseline_character.png"))}, deleteFile = FALSE)
            output$simple_plot = renderImage({return(list(src = "model_images/simple_character.png"))}, deleteFile = FALSE)
            output$large_plot = renderImage({return(list(src = "model_images/large_character.png"))}, deleteFile = FALSE)
        } else if (input$select_fit == "Age") {
            output$baseline_plot = renderImage({return(list(src = "model_images/baseline_age.png"))}, deleteFile = FALSE)
            output$simple_plot = renderImage({return(list(src = "model_images/simple_age.png"))}, deleteFile = FALSE)
            output$large_plot = renderImage({return(list(src = "model_images/large_age.png"))}, deleteFile = FALSE)
        } else if (input$select_fit == "Gender") {
            output$baseline_plot = renderImage({return(list(src = "model_images/baseline_gender.png"))}, deleteFile = FALSE)
            output$simple_plot = renderImage({return(list(src = "model_images/simple_gender.png"))}, deleteFile = FALSE)
            output$large_plot = renderImage({return(list(src = "model_images/large_gender.png"))}, deleteFile = FALSE)
        }
    })
}

# Run the application
shinyApp(ui = ui, server = server)
