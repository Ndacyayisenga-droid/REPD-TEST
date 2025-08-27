public class TestFile9 {
    public static void main(String[] args) {
        TestFile9 test = new TestFile9();
        System.out.println("Beginning network and pattern analysis tests...");
        test.performTests();
        System.out.println("Network and pattern analysis completed successfully.");
    }

    public void performTests() {
        testNetworkingConcepts();
        testDateTimeOperations();
        testMathematicalFunctions();
        testDesignPatterns();
    }

    public void testNetworkingConcepts() {
        simulateNetworkOperations();
        testUrlProcessing();
        simulateHttpRequests();
    }

    public void testDateTimeOperations() {
        java.time.LocalDateTime now = java.time.LocalDateTime.now();
        System.out.println("Current date and time: " + now);
        
        java.time.LocalDate today = java.time.LocalDate.now();
        java.time.LocalDate tomorrow = today.plusDays(1);
        java.time.LocalDate lastWeek = today.minusWeeks(1);
        
        System.out.println("Today: " + today);
        System.out.println("Tomorrow: " + tomorrow);
        System.out.println("Last week: " + lastWeek);
        
        java.time.format.DateTimeFormatter formatter = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        String formattedDateTime = now.format(formatter);
        System.out.println("Formatted: " + formattedDateTime);
        
        calculateTimeDifferences();
        testTimeZones();
    }

    public void testMathematicalFunctions() {
        double[] values = {1.5, 2.7, 3.14, 4.0, 5.5};
        
        for (double value : values) {
            double sqrt = Math.sqrt(value);
            double sin = Math.sin(value);
            double cos = Math.cos(value);
            double log = Math.log(value);
            double exp = Math.exp(value);
            
            System.out.printf("Value: %.2f, Sqrt: %.2f, Sin: %.2f, Cos: %.2f, Log: %.2f, Exp: %.2f%n",
                            value, sqrt, sin, cos, log, exp);
        }
        
        testStatisticalOperations();
        testComplexCalculations();
    }

    public void testDesignPatterns() {
        testSingletonPattern();
        testFactoryPattern();
        testObserverPattern();
        testStrategyPattern();
    }

    private void simulateNetworkOperations() {
        String[] hosts = {"localhost", "google.com", "github.com", "stackoverflow.com"};
        
        for (String host : hosts) {
            boolean isReachable = simulatePing(host);
            System.out.println("Host " + host + " is " + (isReachable ? "reachable" : "unreachable"));
        }
        
        simulateSocketConnection();
    }

    private void testUrlProcessing() {
        String[] urls = {
            "https://www.example.com/path?param=value",
            "http://localhost:8080/api/data",
            "ftp://files.example.com/downloads/",
            "mailto:user@example.com"
        };
        
        for (String url : urls) {
            parseUrl(url);
        }
    }

    private void simulateHttpRequests() {
        String[] methods = {"GET", "POST", "PUT", "DELETE"};
        String[] endpoints = {"/users", "/products", "/orders", "/categories"};
        
        for (String method : methods) {
            for (String endpoint : endpoints) {
                simulateHttpRequest(method, endpoint);
            }
        }
    }

    private void calculateTimeDifferences() {
        java.time.LocalDateTime start = java.time.LocalDateTime.of(2023, 1, 1, 10, 0);
        java.time.LocalDateTime end = java.time.LocalDateTime.of(2023, 12, 31, 18, 30);
        
        java.time.Duration duration = java.time.Duration.between(start, end);
        long days = duration.toDays();
        long hours = duration.toHours();
        long minutes = duration.toMinutes();
        
        System.out.println("Time difference:");
        System.out.println("Days: " + days);
        System.out.println("Hours: " + hours);
        System.out.println("Minutes: " + minutes);
    }

    private void testTimeZones() {
        java.time.ZoneId utc = java.time.ZoneId.of("UTC");
        java.time.ZoneId newYork = java.time.ZoneId.of("America/New_York");
        java.time.ZoneId tokyo = java.time.ZoneId.of("Asia/Tokyo");
        
        java.time.ZonedDateTime utcTime = java.time.ZonedDateTime.now(utc);
        java.time.ZonedDateTime nyTime = utcTime.withZoneSameInstant(newYork);
        java.time.ZonedDateTime tokyoTime = utcTime.withZoneSameInstant(tokyo);
        
        System.out.println("UTC: " + utcTime);
        System.out.println("New York: " + nyTime);
        System.out.println("Tokyo: " + tokyoTime);
    }

    private void testStatisticalOperations() {
        double[] dataset = {12.5, 15.2, 18.7, 22.1, 25.6, 28.9, 32.4, 35.8, 39.2, 42.7};
        
        double mean = calculateMean(dataset);
        double median = calculateMedian(dataset);
        double variance = calculateVariance(dataset, mean);
        double stdDev = Math.sqrt(variance);
        
        System.out.println("Statistical analysis:");
        System.out.printf("Mean: %.2f%n", mean);
        System.out.printf("Median: %.2f%n", median);
        System.out.printf("Variance: %.2f%n", variance);
        System.out.printf("Standard Deviation: %.2f%n", stdDev);
    }

    private void testComplexCalculations() {
        // Matrix operations
        double[][] matrix1 = {{1, 2}, {3, 4}};
        double[][] matrix2 = {{5, 6}, {7, 8}};
        
        double[][] product = multiplyMatrices(matrix1, matrix2);
        System.out.println("Matrix multiplication result:");
        printMatrix(product);
        
        // Polynomial evaluation
        double[] coefficients = {1, 2, 3, 4}; // represents 4x³ + 3x² + 2x + 1
        double x = 2.0;
        double result = evaluatePolynomial(coefficients, x);
        System.out.printf("Polynomial evaluation at x=%.2f: %.2f%n", x, result);
        
        // Numerical integration (trapezoidal rule)
        double integral = trapezoidalIntegration(0, Math.PI, 100);
        System.out.printf("Numerical integration of sin(x) from 0 to π: %.4f%n", integral);
    }

    private void testSingletonPattern() {
        Singleton instance1 = Singleton.getInstance();
        Singleton instance2 = Singleton.getInstance();
        
        System.out.println("Singleton test:");
        System.out.println("Same instance: " + (instance1 == instance2));
        
        instance1.doSomething();
        instance2.doSomething();
    }

    private void testFactoryPattern() {
        AnimalFactory factory = new AnimalFactory();
        
        Animal dog = factory.createAnimal("dog");
        Animal cat = factory.createAnimal("cat");
        Animal bird = factory.createAnimal("bird");
        
        dog.makeSound();
        cat.makeSound();
        bird.makeSound();
    }

    private void testObserverPattern() {
        NewsAgency agency = new NewsAgency();
        NewsChannel channel1 = new NewsChannel("CNN");
        NewsChannel channel2 = new NewsChannel("BBC");
        
        agency.addObserver(channel1);
        agency.addObserver(channel2);
        
        agency.setNews("Breaking news: Design patterns are important!");
        agency.setNews("Update: Observer pattern implemented successfully!");
    }

    private void testStrategyPattern() {
        Calculator calculator = new Calculator();
        
        calculator.setStrategy(new AdditionStrategy());
        int result1 = calculator.execute(5, 3);
        System.out.println("Addition result: " + result1);
        
        calculator.setStrategy(new MultiplicationStrategy());
        int result2 = calculator.execute(5, 3);
        System.out.println("Multiplication result: " + result2);
    }

    // Helper methods
    private boolean simulatePing(String host) {
        // Simulate network ping
        return Math.random() > 0.2; // 80% success rate
    }

    private void simulateSocketConnection() {
        System.out.println("Simulating socket connection...");
        try {
            Thread.sleep(100); // Simulate connection time
            System.out.println("Socket connection established");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void parseUrl(String url) {
        System.out.println("Parsing URL: " + url);
        
        if (url.startsWith("https://")) {
            System.out.println("  Protocol: HTTPS");
        } else if (url.startsWith("http://")) {
            System.out.println("  Protocol: HTTP");
        } else if (url.startsWith("ftp://")) {
            System.out.println("  Protocol: FTP");
        }
        
        // Simple domain extraction
        String domain = extractDomain(url);
        System.out.println("  Domain: " + domain);
    }

    private String extractDomain(String url) {
        String domain = url.replaceAll("^https?://", "").replaceAll("^ftp://", "").replaceAll("^mailto:", "");
        int slashIndex = domain.indexOf('/');
        if (slashIndex != -1) {
            domain = domain.substring(0, slashIndex);
        }
        int atIndex = domain.indexOf('@');
        if (atIndex != -1) {
            domain = domain.substring(atIndex + 1);
        }
        return domain;
    }

    private void simulateHttpRequest(String method, String endpoint) {
        int statusCode = (int) (Math.random() * 300) + 200;
        long responseTime = (long) (Math.random() * 1000) + 50;
        
        System.out.printf("%s %s - Status: %d, Time: %dms%n", method, endpoint, statusCode, responseTime);
    }

    private double calculateMean(double[] data) {
        double sum = 0;
        for (double value : data) {
            sum += value;
        }
        return sum / data.length;
    }

    private double calculateMedian(double[] data) {
        double[] sorted = data.clone();
        java.util.Arrays.sort(sorted);
        
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }

    private double calculateVariance(double[] data, double mean) {
        double sum = 0;
        for (double value : data) {
            sum += Math.pow(value - mean, 2);
        }
        return sum / data.length;
    }

    private double[][] multiplyMatrices(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = b[0].length;
        int common = a[0].length;
        
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    private void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("%8.2f", value);
            }
            System.out.println();
        }
    }

    private double evaluatePolynomial(double[] coefficients, double x) {
        double result = 0;
        for (int i = 0; i < coefficients.length; i++) {
            result += coefficients[i] * Math.pow(x, i);
        }
        return result;
    }

    private double trapezoidalIntegration(double a, double b, int n) {
        double h = (b - a) / n;
        double sum = Math.sin(a) + Math.sin(b);
        
        for (int i = 1; i < n; i++) {
            double x = a + i * h;
            sum += 2 * Math.sin(x);
        }
        
        return (h / 2) * sum;
    }

    // Design Pattern Classes
    static class Singleton {
        private static Singleton instance;
        
        private Singleton() {}
        
        public static synchronized Singleton getInstance() {
            if (instance == null) {
                instance = new Singleton();
            }
            return instance;
        }
        
        public void doSomething() {
            System.out.println("Singleton instance is working");
        }
    }

    interface Animal {
        void makeSound();
    }

    static class Dog implements Animal {
        public void makeSound() {
            System.out.println("Woof! Woof!");
        }
    }

    static class Cat implements Animal {
        public void makeSound() {
            System.out.println("Meow! Meow!");
        }
    }

    static class Bird implements Animal {
        public void makeSound() {
            System.out.println("Tweet! Tweet!");
        }
    }

    static class AnimalFactory {
        public Animal createAnimal(String type) {
            switch (type.toLowerCase()) {
                case "dog": return new Dog();
                case "cat": return new Cat();
                case "bird": return new Bird();
                default: return null;
            }
        }
    }

    interface Observer {
        void update(String news);
    }

    static class NewsAgency {
        private java.util.List<Observer> observers = new java.util.ArrayList<>();
        private String news;
        
        public void addObserver(Observer observer) {
            observers.add(observer);
        }
        
        public void removeObserver(Observer observer) {
            observers.remove(observer);
        }
        
        public void setNews(String news) {
            this.news = news;
            notifyObservers();
        }
        
        private void notifyObservers() {
            for (Observer observer : observers) {
                observer.update(news);
            }
        }
    }

    static class NewsChannel implements Observer {
        private String name;
        
        public NewsChannel(String name) {
            this.name = name;
        }
        
        public void update(String news) {
            System.out.println(name + " received news: " + news);
        }
    }

    interface Strategy {
        int execute(int a, int b);
    }

    static class AdditionStrategy implements Strategy {
        public int execute(int a, int b) {
            return a + b;
        }
    }

    static class MultiplicationStrategy implements Strategy {
        public int execute(int a, int b) {
            return a * b;
        }
    }

    static class Calculator {
        private Strategy strategy;
        
        public void setStrategy(Strategy strategy) {
            this.strategy = strategy;
        }
        
        public int execute(int a, int b) {
            return strategy.execute(a, b);
        }
    }
}
