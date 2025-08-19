import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Hierarchical Reasoning Model (HRM) — PyTorch for Natural-Language Math QA

    Here I will use an HRM text seq2seq model archetecutere to solve complex math problems of the following type: 

    ```csv
    task,answer
    "The value of $y$ varies inversely as $\sqrt x$ and when $x=24$, $y=15$. What is $x$ when $y=3$?",[600]
    ```
    """
    )
    return


@app.cell
def _():
    import torch
    from torch import nn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Defining the HRM archetecture

    The archetecture consisits of the following:\n
    - Input
    - L-model (low level)
    - H-model (high level)
    - Output

    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
