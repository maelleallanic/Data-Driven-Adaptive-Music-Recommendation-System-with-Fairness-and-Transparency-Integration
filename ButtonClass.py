#button class that takes in all the info needed to draw a button
#  and the type of button (simplify button, or clear button)
class Button:
    def __init__(self, topX, topY, bottomX, bottomY, text, fill, outline, 
    outlineWidth, font, fontColor, f):
        self.topX = topX
        self.topY = topY
        self.bottomX = bottomX
        self.bottomY = bottomY
        self.text = text
        self.fill = fill
        self.outline = outline
        self.outlineWidth = outlineWidth
        self.font = font
        self.fontColor = fontColor
        self.f = f


    #takes in self and canvas, draws the button on the canvas
    def drawButton(self, canvas):
        canvas.create_rectangle(self.topX, self.topY, self.bottomX, 
        self.bottomY, fill=self.fill, outline=self.outline, 
        width = self.outlineWidth)
        canvas.create_text((self.topX+self.bottomX)/2, (self.topY+self.bottomY)/2, 
                        anchor = 'c',
                        text=self.text, font=self.font, fill=self.fontColor)
    
    #takes in x,y from mouse and returns true if the user pressed 
    # in the coordinates of the mouse
    #else returns false
    def CheckButtonClicked(self, x, y):
        if( x >=self.topX and x <= self.bottomX 
        and y >= self.topY and y <= self.bottomY):
            return True
        else:
            return False

    #If the button is clicked, it does the action based on the type of button
    def buttonClicked(self, app):
        if self.f != None:
            self.f(app)

#subclass of Button class, takes in the same variables 
# but also the on and off color
#learned how to pass functions 
# from https://www.cs.cmu.edu/~112/notes/notes-functions-redux.html and in lecture
class ToggleButton(Button):
    
    def __init__(self, topX, topY, bottomX, bottomY, text, fillOff, 
    fillOn, outline, outlineWidth, font, fontColor, f):
        self.fillOff = fillOff
        self.fillOn = fillOn
        super().__init__( topX, topY, bottomX, bottomY, text, 
        self.fillOff, outline, outlineWidth, font, fontColor, f)
    
    #takes in app and switches the color of the button
    def buttonClicked(self, app):
        if( self.fill == self.fillOff):
            self.fill = self.fillOn
            self.text = "✓"
        else:
            self.fill = self.fillOff
            self.text = ''

        if self.f != None:
            self.f(app)
