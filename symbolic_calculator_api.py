import io
import base64
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

app = FastAPI(
    title="Symbolic Calculator API",
    description="API for symbolic differentiation, integration, and function plotting",
    version="1.0.0"
)

class CalculationResponse(BaseModel):
    original_expression: str
    result_expression: str
    simplified_result: Optional[str] = None
    latex_original: Optional[str] = None
    latex_result: Optional[str] = None
    plot_image: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

def parse_expression(expr_str: str):
    """Parse a string into a sympy expression with error handling"""
    try:
        transformations = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(expr_str, transformations=transformations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing expression: {str(e)}")

def differentiate(expr, variable=sp.Symbol('x')):
    """Differentiate an expression with respect to the given variable"""
    try:
        return sp.diff(expr, variable)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during differentiation: {str(e)}")

def integrate(expr, variable=sp.Symbol('x'), lower_bound=None, upper_bound=None):
    """Integrate an expression, either definite or indefinite"""
    try:
        if lower_bound is not None and upper_bound is not None:
            return sp.integrate(expr, (variable, lower_bound, upper_bound))
        return sp.integrate(expr, variable)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during integration: {str(e)}")

def create_plot(expr, result, mode, x_min=-10, x_max=10, points=1000):
    """Create a plot of the original function and its derivative/integral and return as base64 image"""
    x = sp.Symbol('x')
    try:
        # Convert sympy expressions to numpy functions
        f = sp.lambdify(x, expr, modules=['numpy'])
        g = sp.lambdify(x, result, modules=['numpy'])
        
        # Create x values
        X = np.linspace(x_min, x_max, points)
        
        # Evaluate functions, handling potential complex values
        Y_f = np.array([complex(f(val)).real for val in X])
        Y_g = np.array([complex(g(val)).real for val in X])
        
        # Create a new figure
        fig = Figure(figsize=(10, 8))
        
        # Create subplots
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        
        # Plot original function
        ax1.plot(X, Y_f, label='Original Function', color='blue', linewidth=2)
        ax1.set_title(f'Original Function')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True)
        ax1.legend()
        
        # Plot result
        ax2.plot(X, Y_g, label=f'{mode.title()} Result', color='red', linewidth=2)
        ax2.set_title(f'{mode.title()} Result')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(True)
        ax2.legend()
        
        fig.tight_layout()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return plot_base64
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during plotting: {str(e)}")

@app.get("/", tags=["Info"])
def root():
    """Root endpoint with API information"""
    return {
        "name": "Symbolic Calculator API",
        "version": "1.0.0",
        "endpoints": {
            "/differentiate": "Differentiate a mathematical expression",
            "/integrate": "Integrate a mathematical expression",
            "/docs": "API documentation"
        }
    }

@app.get("/differentiate", response_model=CalculationResponse, responses={400: {"model": ErrorResponse}}, tags=["Calculus"])
def api_differentiate(
    expression: str = Query(..., description="Mathematical expression in x (e.g., 'x**2 + sin(x)')", example="x**2 + sin(x)"),
    variable: str = Query("x", description="Variable to differentiate with respect to"),
    generate_plot: bool = Query(True, description="Whether to generate and return a plot"),
    x_min: float = Query(-10, description="Minimum x value for plotting"),
    x_max: float = Query(10, description="Maximum x value for plotting"),
    points: int = Query(1000, description="Number of points to plot")
):
    """
    Differentiate a mathematical expression with respect to a variable.
    
    Returns the derivative and optionally a plot of both the original function and its derivative.
    """
    x = sp.Symbol(variable)
    expr = parse_expression(expression)
    result = differentiate(expr, x)
    simplified = sp.simplify(result)
    
    response = CalculationResponse(
        original_expression=str(expr),
        result_expression=str(result),
        simplified_result=str(simplified) if simplified != result else None,
        latex_original=sp.latex(expr),
        latex_result=sp.latex(result)
    )
    
    if generate_plot:
        response.plot_image = create_plot(expr, result, "differentiate", x_min, x_max, points)
    
    return response

@app.get("/integrate", response_model=CalculationResponse, responses={400: {"model": ErrorResponse}}, tags=["Calculus"])
def api_integrate(
    expression: str = Query(..., description="Mathematical expression in x (e.g., 'x**2 + sin(x)')", example="x**2 + sin(x)"),
    variable: str = Query("x", description="Variable to integrate with respect to"),
    lower_bound: Optional[float] = Query(None, description="Lower bound for definite integral"),
    upper_bound: Optional[float] = Query(None, description="Upper bound for definite integral"),
    generate_plot: bool = Query(True, description="Whether to generate and return a plot"),
    x_min: float = Query(-10, description="Minimum x value for plotting"),
    x_max: float = Query(10, description="Maximum x value for plotting"),
    points: int = Query(1000, description="Number of points to plot")
):
    """
    Integrate a mathematical expression with respect to a variable.
    
    Supports both indefinite integration and definite integration with specified bounds.
    Returns the integral and optionally a plot of both the original function and its integral.
    """
    # Validate that both bounds are provided if one is
    if (lower_bound is not None and upper_bound is None) or (lower_bound is None and upper_bound is not None):
        raise HTTPException(status_code=400, detail="Both lower_bound and upper_bound must be provided for definite integrals")
    
    x = sp.Symbol(variable)
    expr = parse_expression(expression)
    result = integrate(expr, x, lower_bound, upper_bound)
    simplified = sp.simplify(result)
    
    response = CalculationResponse(
        original_expression=str(expr),
        result_expression=str(result),
        simplified_result=str(simplified) if simplified != result else None,
        latex_original=sp.latex(expr),
        latex_result=sp.latex(result)
    )
    
    # Only generate plot for indefinite integrals
    if generate_plot and lower_bound is None and upper_bound is None:
        response.plot_image = create_plot(expr, result, "integrate", x_min, x_max, points)
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
