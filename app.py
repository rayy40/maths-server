from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from sympy import Matrix, parse_expr, latex, symbols, Eq, solve, factor, Symbol
from sympy.matrices.common import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
import numpy as np
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RootsRequest(BaseModel):
    equation: str
    
class SimultaneousEquation(BaseModel):
    equation: Dict[str, str]
    
class MatrixRequestHistory(BaseModel):
    matrixHistory: Dict[str, List[List[str]]]
    equation: str

class MatrixRequest(BaseModel):
    matrix: List[List[str]] = []
    
    
def convert_to_katex(latex_expression):
    latex_expression = latex_expression.replace(r'\\', r' \\newline ')
    latex_expression = latex_expression.replace(r'\left[\begin{matrix}', r'\\begin{bmatrix}')
    latex_expression = latex_expression.replace(r'\end{matrix}\right]', r'\\end{bmatrix}')  
    
    return latex_expression  

    
@app.post("/api/matrix/square")
async def matrix_square(request: Request, data: MatrixRequest):
    
    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    matrix_squared = np.round(np.matmul(matrix, matrix), decimals=2)
    
    return {"result": matrix_squared.tolist()}


@app.post("/api/matrix/inverse")
async def matrix_square(request: Request, data: MatrixRequest):
    try:
        matrix = np.array([[int(num) for num in row] for row in data.matrix])
        matrix_inverse = np.round(np.linalg.inv(matrix), decimals=2)
        return {"result": matrix_inverse.tolist()}
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/matrix/trace")
async def matrix_square(request: Request, data: MatrixRequest):
    
    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    matrix_trace = np.trace(matrix)
    
    if isinstance(matrix_trace, float):
        matrix_trace = round(matrix_trace, 3)
    
    return {"result": str(matrix_trace)}


@app.post("/api/matrix/determinant")
async def matrix_square(request: Request, data: MatrixRequest):
    
    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    matrix_det = np.linalg.det(matrix)
    
    if isinstance(matrix_det, float):
        matrix_det = round(matrix_det, 3)
    
    return {"result": str(matrix_det)}


@app.post("/api/matrix/rref")
async def matrix_rref(request: Request, data: MatrixRequest):
    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    sympy_matrix = Matrix(matrix)
    
    rref_matrix = sympy_matrix.rref()[0]
    
    np_matrix = np.round(np.array(rref_matrix.tolist(), dtype=float), decimals=2)
    
    return {"result": np_matrix.tolist()}


@app.post("/api/matrix/transpose")
async def matrix_square(request: Request, data: MatrixRequest):
    
    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    matrix_transpose = matrix.T
    
    return {"result": matrix_transpose.tolist()}


@app.post("/api/matrix/eigen")
async def matrix_eigen_value(request: Request, data: MatrixRequest):

    matrix = np.array([[int(num) for num in row] for row in data.matrix])
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    eigenvalues = np.round(eigenvalues, decimals=2)
    eigenvectors = np.round(eigenvectors, decimals=2)

    # Convert complex values to strings
    eigenvalues = [str(value) for value in eigenvalues]
    eigenvectors = [[str(value) for value in vector] for vector in eigenvectors]
    
    return {"value": eigenvalues, "vector": eigenvectors}



@app.post("/api/matrix/equation")
async def matrix_equation(request: Request, data: MatrixRequestHistory):
    try:  
        matrixHistory = data.matrixHistory
        equation = data.equation
        
        variable_symbols = set()
        operation_symbols = set()
        index = 0
        
        while index < len(equation):
            char = equation[index]
            
            if char.isalpha():
                if char == 'T':
                    index += 1  # Skip only the current character for 'T'
                    continue
                
                if equation[index:index+3] == 'det':
                    operation_symbols.add('det')
                    index += 2  
                elif equation[index:index+5] == 'trace':
                    operation_symbols.add('trace')
                    index += 4 
                elif equation[index:index+4] == 'rref':
                    operation_symbols.add('rref')
                    index += 3 
                else:
                    variable_symbols.add(char)
            
            index += 1
        
        matrix_values = {symbol: Matrix(matrix) for symbol, matrix in matrixHistory.items()}
        
        expression = equation.strip("'").replace('^', '**').replace('{', '(').replace('}', ')').replace('**T', '.transpose()')
        
        substitution = {}
        for symbol in variable_symbols:
            symbol_power = symbol + '^2'
            symbol_inverse = symbol + '^(-1)'
        
            if symbol_power in equation:
                matrix_values[symbol_power] = matrix_values[symbol] ** 2
            
            if symbol_inverse in equation:
                matrix_values[symbol_inverse] = matrix_values[symbol].inv()
                
            if 'det' in operation_symbols:
                determinant = round(matrix_values[symbol].det(), 2)
                expression = expression.replace('det({})'.format(symbol), str(determinant))
                
            if 'trace' in operation_symbols:
                trace = round(matrix_values[symbol].trace(), 2)
                expression = expression.replace('trace({})'.format(symbol), str(trace))
                
            # if 'rref' in operation_symbols:
            #     rref = matrix_values[symbol].rref()
            #     expression = expression.replace('rref({})'.format(symbol), rref)
                
            substitution[symbol] = matrix_values[symbol]
            
        expression_expr = parse_expr(expression)
        
        substituted_expression = expression_expr.subs(substitution)

        result = substituted_expression.evalf()
        
        if isinstance(result, Matrix):
            np_matrix = np.around(np.array(result.tolist(), dtype=float), decimals=2)
            result_list = np_matrix.tolist()
            result_type = "matrix"
        else:
            result_type = "scalar"
        
        latex_expression = latex(result_list if result_type == "matrix" else result)
        
        latex_expression = latex_expression.replace(r'\\', r' \newline ')
        latex_expression = latex_expression.replace(r'\left[\begin{matrix}', r'\begin{bmatrix}')
        latex_expression = latex_expression.replace(r'\end{matrix}\right]', r'\end{bmatrix}')
        
        result_output = {
            "latex": latex_expression,
            "type": result_type
        }
        return result_output
    
    except (ShapeError, NonSquareMatrixError, NonInvertibleMatrixError)  as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/roots/equation-solver")
async def equation_solver(request: Request, data: RootsRequest):
    
    expression = data.equation.strip("'").replace('^', '**').replace('{', '(').replace('}', ')').replace("\\", "")
    
    if not expression.startswith('-') and not expression.startswith('+'):
        expression = '+' + expression
    
    equation = expression.replace('x', '*x')
    equation = equation.split("=")

    parsed_equation = parse_expr(equation[0])
    
    x = symbols('x')
    
    equation = Eq(parsed_equation, float(equation[1]))
    
    solutions = solve(equation, x)
    
    factors = factor(parsed_equation)

    result = [];
    
    for solution in solutions:
        latex_roots = latex(round(solution, 3))
        result.append(latex_roots)
    
    latex_factors = latex(factors);
    
    return {"roots": result, "factors": latex_factors}


@app.post("/api/roots/simultaneous-equation")
async def simultaneous_equation(request: Request, data: SimultaneousEquation):
    equations = data.equation
    
    parsed_eqs = []
    symbols = []
    for eq in equations.values():
        eq = eq.replace(' ', '')  
        lhs, rhs = eq.split('=')
        lhs = lhs.replace('x', '*x').replace('y', '*y').replace('z','*z').replace('u','*u')  
        lhs = '(' + lhs.replace('=', '-') + ')'  
        parsed_eqs.append(lhs + '-' + rhs)
    
    combined_equations = ' '.join(parsed_eqs)
    variables = set(re.findall(r'[a-zA-Z]+', combined_equations))
        
    sympy_eqs = [parse_expr(eq) for eq in parsed_eqs]

    for variable in variables:
        symbol = Symbol(variable)
        symbols.append(symbol)
        
    result = solve(sympy_eqs, symbols)
    
    result_dict = {str(symbol): str(solution) for symbol, solution in result.items()}
    
    print(result_dict)
    
    return {"result": result_dict}