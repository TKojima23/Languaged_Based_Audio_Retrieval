from app.engine import SliceSearchEngine
from app.gui import App


def main():
    engine = SliceSearchEngine()
    app = App(engine)
    app.mainloop()


if __name__ == "__main__":
    main()

