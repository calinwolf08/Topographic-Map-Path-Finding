from threading import Event, Thread

class ChildProgram:  
    def __init__(self, name, ready=None):
        self.ready = ready
        self.name = name

    def connect(self):
        # lets make connection, expensive
        print("connecting " + self.name + "... ")

        # then fire the ready event
        self.ready.set()

    def some_logic(self):
        # do something with self.connection
        print("something with " + self.name)

ready = Event()  
ready2 = Event()  
ready3 = Event()  
ready4 = Event()  
program = ChildProgram("test1", ready)
program2 = ChildProgram("test2", ready2)
program3 = ChildProgram("test3", ready3)
program4 = ChildProgram("test4", ready4)

# configure & start thread
thread = Thread(target=program.connect)  
thread.start()
thread2 = Thread(target=program2.connect)  
thread2.start()
thread3 = Thread(target=program3.connect)  
thread3.start()
thread4 = Thread(target=program4.connect)  
thread4.start()

ready4.wait()
program4.some_logic()

ready.wait()
program.some_logic()

ready3.wait()
program3.some_logic()

ready2.wait()
program2.some_logic()

