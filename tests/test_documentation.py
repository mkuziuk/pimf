from html.parser import HTMLParser
from pathlib import Path


def test_documentation_examples_and_local_links():
    class Documentation(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
            self.links = []
            self.examples = []
            self.in_python = False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                assert attrs["id"] not in self.ids
                self.ids.add(attrs["id"])
            for attr in ("href", "src"):
                if attr in attrs:
                    self.links.append(attrs[attr])
            if tag == "code" and attrs.get("class") == "language-python":
                self.in_python = True
                self.examples.append("")

        def handle_endtag(self, tag):
            if tag == "code":
                self.in_python = False

        def handle_data(self, data):
            if self.in_python:
                self.examples[-1] += data

    root = Path(__file__).parents[1] / "docs"
    document = Documentation()
    document.feed((root / "index.html").read_text())
    assert document.examples
    for link in document.links:
        if link.startswith("#"):
            assert link[1:] in document.ids, link
        elif not link.startswith("https://"):
            assert (root / link).is_file(), link
    namespace = {}
    for index, example in enumerate(document.examples):
        exec(compile(example, f"docs/index.html example {index + 1}", "exec"), namespace)
