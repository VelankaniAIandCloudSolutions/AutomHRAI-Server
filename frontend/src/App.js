import logo from './logo.svg';
import './App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Home from "../src/pages/Home"

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
        <Route path='/' element={<Home />} />
        </Routes>
        </BrowserRouter>
    </div>
  );
}

export default App;
