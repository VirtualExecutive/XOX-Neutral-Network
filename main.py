import numpy as np

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid'in türevi
def sigmoid_derivative(x):
    return x * (1 - x)

# Yapay Sinir Ağı sınıfı
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Giriş, gizli ve çıkış katmanlarının boyutları
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Rastgele ağırlıkların başlatılması
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
    
    def forward(self, inputs):
        # Girişten gizli katmana geçiş
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = sigmoid(hidden_inputs)
        
        # Gizli katmandan çıkış katmanına geçiş
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = sigmoid(final_inputs)
        
        return final_outputs
    
    def train(self, inputs, targets, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            # İleri besleme
            hidden_inputs = np.dot(inputs, self.weights_input_hidden)
            hidden_outputs = sigmoid(hidden_inputs)
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
            final_outputs = sigmoid(final_inputs)
            
            # Hata hesaplanması
            output_errors = targets - final_outputs
            
            # Geri yayılım
            output_delta = output_errors * sigmoid_derivative(final_outputs)
            hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_errors * sigmoid_derivative(hidden_outputs)
            
            # Ağırlık güncellemesi
            self.weights_hidden_output += np.dot(hidden_outputs.T, output_delta) * learning_rate
            self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate

# Örnek kullanım
if __name__ == "__main__":
    # Örnek giriş verileri ve hedef çıktılar
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Yapay Sinir Ağı oluşturulması
    neural_network = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    
    while True:
        # Ağın eğitilmesi
        neural_network.train(inputs=X, targets=y, num_epochs=100000, learning_rate=0.1)
        
        # Test
        test_input = np.array([[0, 0]])
        print("0  XOR  0 = ", neural_network.forward(test_input))
        
        test_input = np.array([[0, 1]])
        print("0  XOR  1 = ", neural_network.forward(test_input))
        
        test_input = np.array([[1, 0]])
        print("1  XOR  0 = ", neural_network.forward(test_input))
        
        test_input = np.array([[1, 1]])
        print("1  XOR  1 = ", neural_network.forward(test_input))
        input("")