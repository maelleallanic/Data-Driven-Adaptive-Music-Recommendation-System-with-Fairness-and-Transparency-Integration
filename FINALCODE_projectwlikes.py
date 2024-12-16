# ## UI MUSIC REC APP ##
from cmu_112_graphics import *
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import zipfile








# IMPORTANT INSTALLS
# pip install cmu-112-graphics pandas scikit-learn
# python.exe -m pip install --upgrade pip
# pip install scikit-learn
# pip install pandas scikit-learn
















def appStarted(app):
  app.currentScreen = 'home'
  app.musicData = None
  app.recommendations = []
  app.knnResults = None
  app.currentRecIndex = 0
  app.recListLen = 0
  app.songList = []
  app.logoImage = app.loadImage('headphones.png')
  app.logoImage = app.scaleImage(app.logoImage, 0.3)
  app.final_recommendations = []
  app.data_incl_liked = []
  app.X = []
  app.buttonClickedLikedSongs = []
  app.sampleImage = app.loadImage('samplesongdata.jpg')








def loadMusicInput(filename, app):
  with open(filename, "r", encoding="utf-8") as f:
     fileString = f.read()








 # parse csv file
  app.songList = []
  for line in fileString.strip().splitlines()[1:]:
      songData = line.split(',')
      songDict = {
          'title': songData[0].strip(),
          'artist': songData[1].strip()
      }
      app.songList.append(songDict)
  return app.songList




def uploadMusicButtonAction(app):
  app.currentScreen = 'upload'
  # load data
  # app.musicData = loadMusicInput("hardcoded_songs_sample.csv", app)
  app.recommendations = knnAlg(app)
  app.currentScreen = 'recommendations'
  return




def knnAlg(app):
 
   print("training in process")


   zip_file_path = './archive.zip'
   extract_folder = './archive/'

   if not os.path.exists(extract_folder):
        print("Unzipping the archive...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print("Unzipping complete.")
   else:
        print("Archive already unzipped.")

    # Load the data
   data_file_path = os.path.join(extract_folder, 'spotify_data.csv')
   if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Expected file {data_file_path} not found after unzipping.")

   print("Loading data...")
   data = pd.read_csv(data_file_path)

   # selected features
   features = ['danceability', 'energy', 'tempo', 'valence', 'acousticness']
 
   # Normalize features
   scaler = StandardScaler()
   X = scaler.fit_transform(data[features])
   app.X = X
 
   # import initial liked songs
   liked_songs = pd.read_csv("initial_liked_songs.csv")  # Load liked songs
   app.musicData = liked_songs



   if (len(app.buttonClickedLikedSongs) > 0):
        new_liked_songs_df = pd.DataFrame(app.buttonClickedLikedSongs, columns=['track_name', 'artist_name'])

        liked_songs = pd.concat([liked_songs, new_liked_songs_df]).drop_duplicates()


 
   app.musicData = liked_songs
   liked_song_set = set(zip(liked_songs['artist_name'], liked_songs['track_name']))
 
   print("training in process2")
   all_songs_zipped = zip(data['artist_name'], data['track_name'])
   data['liked'] = [(1 if (artist, track) in liked_song_set else 0) for (artist, track) in all_songs_zipped]
   app.data_incl_liked = data
 
   print("training in process3")
   # train knn regressor
   knn_regressor = neighbors.KNeighborsRegressor(n_neighbors=15, metric='euclidean')
   knn_regressor.fit(X, data['liked'])
 
   # predict similarity scores
   data['similarity_score'] = knn_regressor.predict(X)
  


   print("training in process4")
   artist_info = pd.read_csv("Empirical Musicology Review Popular-music Artist Demographic Database - EMR (MBB, RS200, UG).csv")
   data_with_artist_info = data.merge(
       artist_info,
       left_on='artist_name',
       right_on='artist',
       how='left'
   )


   # fill missing values for demographic columns with 0
   diversity_columns = ['nonmale', 'non-cis', 'race', 'ethnicity']
   data_with_artist_info[diversity_columns] = data_with_artist_info[diversity_columns].fillna(0)

   diverse_artists = data_with_artist_info[
       (data_with_artist_info['nonmale'] == 1) |
       (data_with_artist_info['non-cis'] == 1) |
       (data_with_artist_info['race'] == 1) |
       (data_with_artist_info['ethnicity'] == 1)
   ]




   print("training in process5")
   diverse_artists = diverse_artists.copy()
   diverse_artists['similarity_score'] = knn_regressor.predict(scaler.transform(diverse_artists[features]))


   # select top similarity-based and diverse recommendations
   similarity_top = data_with_artist_info.sort_values(by='similarity_score', ascending=False).head(15)
   diversity_top = diverse_artists.sort_values(by='similarity_score', ascending=False).head(15)



   # combing similarity and diversity recomendations
   final_recommendations = pd.concat([similarity_top, diversity_top]).drop_duplicates()


   # calculate combined score
   final_recommendations['combined_score'] = (
       0.7 * final_recommendations['similarity_score'] +
       0.3 * (final_recommendations[diversity_columns].sum(axis=1))
   )

   print("training in process6")


   # get combined score and get final ranking
   app.final_recommendations = final_recommendations.sort_values(by='combined_score', ascending=False).head(25)


   print("Top hybrid recommendations (similarity + diversity):")
   print(final_recommendations[['artist_name', 'track_name', 'similarity_score', 'combined_score']])
   result = [[row['track_name'], row['artist_name']] for _, row in final_recommendations.iterrows()]
   app.recListLen = 25


   print("done")
   knnDisplayImage(app)
   return result





def knnDisplayImage(app):
   
   # 2d projection of the feature space for visualization using PCA

   if ( (len(app.data_incl_liked) == 0) or (len(app.final_recommendations) == 0)):
       print("cant make graph, no data yet")
       return
   data = app.data_incl_liked
   final_recommendations = app.final_recommendations


   pca = PCA(n_components=2)
   X_2d = pca.fit_transform(app.X)

   # scatter plot with all songs
   plt.figure(figsize=(10, 8))


   # non-liked songs (light blue, transparent) 
   not_liked_indices = data[data['liked'] == 0].index


   plt.scatter(
       X_2d[not_liked_indices, 0],
       X_2d[not_liked_indices, 1],
       c='lightblue',
       s=50,
       alpha=0.3,
       label='Non-Liked Songs'
   )


   # liked songs (red, opaque)
   liked_indices = data[data['liked'] == 1].index

   plt.scatter(
       X_2d[liked_indices, 0],
       X_2d[liked_indices, 1],
       c='red',
       s=200,
       alpha=1.0,
       label='Liked Songs'
   )



   # recommended songs (green) 
   recommended_indices = final_recommendations.index


   plt.scatter(
       X_2d[recommended_indices, 0],
       X_2d[recommended_indices, 1],
       c='green',
       s=150,
       alpha=0.9,
       label='Recommended Songs'

   )


   # add song name for liked songs
   for i in liked_indices:
       plt.annotate(
           data.loc[i, 'track_name'],
           (X_2d[i, 0], X_2d[i, 1]),
           fontsize=9,
           alpha=0.7
       )


   plt.title("Song Recommendations Based on Liked Songs", fontsize=14)
   plt.xlabel("PCA Component 1", fontsize=12)
   plt.ylabel("PCA Component 2", fontsize=12)
   plt.legend(loc='best')
   plt.grid(True)
   plt.show()
   return












def likeRecButtonAction(app):
  #recommend another song similar to this one
  song = app.recommendations[app.currentRecIndex]
  app.buttonClickedLikedSongs.append(song)
  print(app.buttonClickedLikedSongs)




  if app.currentRecIndex + 1 < app.recListLen:
      app.currentRecIndex = app.currentRecIndex + 1
  else:
      app.currentScreen = 'final'
  return












def dislikeRecButtonAction(app):
  #delete the song from the list
  if app.currentRecIndex + 1 < app.recListLen:
      app.recommendations.pop(app.currentRecIndex)
      app.recListLen -= 1
  else:
      app.recommendations.pop(app.currentRecIndex)
      app.recListLen -= 1
     
      app.currentScreen = 'final'
  return




def getMoreRecsButtonAction(app):
   # get more recommendations
   uploadMusicButtonAction(app)
   return










def redrawAll(app, canvas):
  canvas.create_rectangle(0, 0, app.width, app.height, fill="lightblue", outline="")
  canvas.create_rectangle(20, 20, app.width - 20, app.height - 20, fill="white", outline="")
  if app.currentScreen == 'home':
      canvas.create_text(app.width//2, app.height//3 - 100, text="Welcome to the Music Recommendation App!", fill="black", font="Script 22 bold")
      canvas.create_image(app.width//2, app.height//3 + 20, image=ImageTk.PhotoImage(app.logoImage))








      #instructions button
      instructionsButtonTopY = app.height//2
      instructionsButtonBottomY = instructionsButtonTopY + 40
      canvas.create_rectangle(app.width//2 - 70, instructionsButtonTopY, app.width//2 + 70, instructionsButtonBottomY, fill="black")
      canvas.create_text(app.width//2, instructionsButtonTopY + 20, text="Instructions", fill="white")








      #upload music data button
      buttonTopY = instructionsButtonBottomY + 20
      buttonBottomY = buttonTopY + 40
      canvas.create_rectangle(app.width//2 - 70, buttonTopY, app.width//2 + 70, buttonBottomY, fill="black")
      canvas.create_text(app.width//2, buttonTopY + 20, text="Upload Music Data", fill="white")








      #transparency/fairness button
      tfButtonTopY = buttonBottomY + 20
      tfButtonBottomY = tfButtonTopY + 40
      canvas.create_rectangle(app.width//2 - 80, tfButtonTopY, app.width//2 + 80, tfButtonBottomY, fill="black")
      canvas.create_text(app.width//2, tfButtonTopY + 20, text="Transparency/Fairness", fill="white")








  elif app.currentScreen == 'instructions':
      #instructions text
      canvas.create_text(app.width//2, app.height//4 - 20, text="Instructions", font="Arial 20 bold", fill="black")
      canvas.create_text(app.width//2, app.height//3 + 20, text="To upload data, make a spreadsheet with 3 columns", fill="black", font="Arial 12")
    #   canvas.create_image(app.width//2, app.height//3 + 250, image=ImageTk.PhotoImage(app.sampleImage))
      canvas.create_text(app.width//2, app.height//3 + 40, text="(Song Number, Song Name, Song Artist). Then, download it as a .csv file!", fill="black", font="Arial 12")
      canvas.create_text(app.width//2, app.height//3 + 100, text="Once you press Upload Music Data, the model will begin training, and a", fill="black", font="Arial 12")
      canvas.create_text(app.width//2, app.height//3 + 120, text="K-Nearest Neighbors graph will pop up. The KNN graph uses your input", fill="black", font="Arial 12")
      canvas.create_text(app.width//2, app.height//3 + 140, text="song data to create a visual representation of the algorithm's", fill="black", font="Arial 12")
      canvas.create_text(app.width//2, app.height//3 + 160, text="clusters of the recommended, liked, and non-liked songs.", fill="black", font="Arial 12")
      canvas.create_text(app.width//2, app.height//3 + 220, text="Make sure you close the graph to continue.", fill="black", font="Arial 12")








      #back button
      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      canvas.create_rectangle(app.width//2 - 50, backButtonTopY, app.width//2 + 50, backButtonBottomY, fill="black")
      canvas.create_text(app.width//2, backButtonTopY + 20, text="Back", fill="white")
















  elif app.currentScreen == 'transparency':
      canvas.create_text(app.width//2, app.height//4, text="Transparency & Fairness", font="Arial 20 bold", fill="black")
      canvas.create_text(app.width//2, app.height//3, text="This algorithm takes the song and artist data you upload,", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 20, text="and compares it on different metrics (genre, danceability,", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 40, text="energy, mood) to other songs in our dataset.", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 80, text="Then, we'll give you some recommendations of similar", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 100, text="songs from artists with different backgrounds!", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//2 + 140, text="Your uploaded song data and your recommendations", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//2 + 160, text="will not be saved by this app or our algorithm.", fill="black", font="Arial 16")
















          #see how this works button
      seeHowButtonTopY = app.height//3 + 140
      seeHowButtonBottomY = seeHowButtonTopY + 40
      canvas.create_rectangle(app.width//2 - 70, seeHowButtonTopY, app.width//2 + 70, seeHowButtonBottomY, fill="black")
      canvas.create_text(app.width//2, seeHowButtonTopY + 20, text="See How This Works!", fill="white")
   
      #back button, go back to home screen
      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      canvas.create_rectangle(app.width//2-50, backButtonTopY, app.width//2+50, backButtonBottomY, fill="black")
      canvas.create_text(app.width//2, backButtonTopY + 20, text="Back", fill="white")
  elif app.currentScreen == 'algorithmExplanation':
      canvas.create_text(app.width//2, app.height//4, text="Algorithm Explanation", font="Arial 20 bold", fill="black")
      canvas.create_text(app.width//2, app.height//3, text="Our recommendation system uses K-Nearest Neighbors", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 20, text="to group songs by various metrics. This allows us to find songs", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 40, text="that share similar characteristics with your song data.", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 80, text="Each cluster represents a distinct musical 'mood' or 'genre',", fill="black", font="Arial 16")
      canvas.create_text(app.width//2, app.height//3 + 100, text="which helps us recommend songs that you might like!", fill="black", font="Arial 16")
   
      #back to transparency screen
      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      canvas.create_rectangle(app.width//2-50, backButtonTopY, app.width//2+50, backButtonBottomY, fill="black")
      canvas.create_text(app.width//2, backButtonTopY + 20, text="Back", fill="white")
  elif app.currentScreen == 'recommendations':
      if app.recommendations:
          song, artist = app.recommendations[app.currentRecIndex]
          canvas.create_text(app.width//2, app.height//3, text=f"Recommendation:", font="Script 18 bold underline")
          canvas.create_text(app.width//2, app.height//3 + 40, text=f"{song} by {artist}", fill="black", font="Script 20")
       
          canvas.create_rectangle(app.width//2-50, app.height//2, app.width//2+50, app.height//2+40, fill="black")
          canvas.create_text(app.width//2, app.height//2+20, text="Like", fill="white")
          canvas.create_rectangle(app.width//2-50, app.height//2+60, app.width//2+50, app.height//2+100, fill="black")
          canvas.create_text(app.width//2, app.height//2+80, text="Dislike", fill="white")
  elif app.currentScreen == 'final':
      canvas.create_text(app.width//2, app.height//9, text="Your Recommendations:", font="Arial 18 bold", fill="black")
      for i, (song, artist) in enumerate(app.recommendations):
          canvas.create_text(app.width//2, app.height//7 + i * 20, text=f"{song} by {artist}", fill="black")
      # image
      #canvas.create_image(app.width//2, app.height * 0.85, image=ImageTk.PhotoImage(app.logoImage)) # logoImage to be changed to knn graph image
      canvas.create_text(app.width//2, app.height - app.height//9, text="Get More Recommendations!", font="Arial 18 bold", fill="black")
     














def mousePressed(app, event):
  if app.currentScreen == 'home':
      #instructions button
      instructionsButtonTopY = app.height//2
      instructionsButtonBottomY = instructionsButtonTopY + 40
      if (app.width//2 - 70 < event.x < app.width//2 + 70) and (instructionsButtonTopY < event.y < instructionsButtonBottomY):
          app.currentScreen = 'instructions'








      # upload music data button
      buttonTopY = instructionsButtonBottomY + 20
      buttonBottomY = buttonTopY + 40
      if (app.width//2 - 70 < event.x < app.width//2 + 70) and (buttonTopY < event.y < buttonBottomY):
          uploadMusicButtonAction(app)








      #transparency/fairness button
      tfButtonTopY = buttonBottomY + 20
      tfButtonBottomY = tfButtonTopY + 40
      if (app.width//2 - 80 < event.x < app.width//2 + 80) and (tfButtonTopY < event.y < tfButtonBottomY):
          app.currentScreen = 'transparency'








  elif app.currentScreen == 'instructions':
      #back button
      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      if (app.width//2 - 50 < event.x < app.width//2 + 50) and (backButtonTopY < event.y < backButtonBottomY):
          app.currentScreen = 'home'
       
  elif app.currentScreen == 'transparency':
















      seeHowButtonTopY = app.height//3 + 140
      seeHowButtonBottomY = seeHowButtonTopY + 40
      if (app.width//2 - 70 < event.x < app.width//2 + 70) and (seeHowButtonTopY < event.y < seeHowButtonBottomY):
          app.currentScreen = 'algorithmExplanation'
















      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      if (app.width//2 - 50 < event.x < app.width//2 + 50) and (backButtonTopY < event.y < backButtonBottomY):
          app.currentScreen = 'home'
















  elif app.currentScreen == 'algorithmExplanation':
      #back button
      backButtonTopY = app.height - 80
      backButtonBottomY = backButtonTopY + 40
      if (app.width//2 - 50 < event.x < app.width//2 + 50) and (backButtonTopY < event.y < backButtonBottomY):
          app.currentScreen = 'transparency'
















  elif app.currentScreen == 'recommendations':
















      if (app.width//2-50 < event.x < app.width//2+50) and (app.height//2 < event.y < app.height//2+40):
          likeRecButtonAction(app)
      elif (app.width//2-50 < event.x < app.width//2+50) and (app.height//2+60 < event.y < app.height//2+100):
          dislikeRecButtonAction(app)




  elif app.currentScreen == 'final':
     
      if (app.width//2-50 < event.x < app.width//2+50) and (app.height - app.height//9 - 50 < event.y < app.height - app.height//9 + 50):
          getMoreRecsButtonAction(app)
















runApp(width=600, height=800)

