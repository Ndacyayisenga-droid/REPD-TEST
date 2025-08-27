package test;

import java.util.*;
import java.util.stream.*;
import java.math.*;
import java.security.*;
import java.text.*;
import java.util.concurrent.*;
import java.util.function.*;

public class TestFile2 {
    private String identifier;
    private BigDecimal balance;
    private Set<String> tags;
    private Queue<String> operations;

    public TestFile2(String identifier) {
        this.identifier = identifier;
        this.balance = BigDecimal.ZERO;
        this.tags = new HashSet<>();
        this.operations = new PriorityQueue<>();
    }

    public static void main(String[] args) {
        TestFile2 test = new TestFile2("TestInstance2");
        test.executeAllTests();
    }

    public void executeAllTests() {
        testMathematicalOperations();
        testStringProcessing();
        testCollectionManipulation();
        testFunctionalProgramming();
        testCryptographicOperations();
        testConcurrentOperations();
        testDataValidation();
        testPerformanceMeasurement();
    }

    public void testMathematicalOperations() {
        System.out.println("=== Mathematical Operations ===");
        
        // BigDecimal calculations
        BigDecimal num1 = new BigDecimal("123.456789");
        BigDecimal num2 = new BigDecimal("987.654321");
        
        BigDecimal sum = num1.add(num2);
        BigDecimal difference = num2.subtract(num1);
        BigDecimal product = num1.multiply(num2);
        BigDecimal quotient = num2.divide(num1, 10, RoundingMode.HALF_UP);
        
        System.out.println("Sum: " + sum);
        System.out.println("Difference: " + difference);
        System.out.println("Product: " + product);
        System.out.println("Quotient: " + quotient);
        
        // Complex number operations
        ComplexNumber c1 = new ComplexNumber(3, 4);
        ComplexNumber c2 = new ComplexNumber(1, 2);
        
        ComplexNumber complexSum = c1.add(c2);
        ComplexNumber complexProduct = c1.multiply(c2);
        double magnitude = c1.magnitude();
        
        System.out.println("Complex sum: " + complexSum);
        System.out.println("Complex product: " + complexProduct);
        System.out.println("Magnitude of c1: " + magnitude);
        
        // Statistical calculations
        double[] data = {1.2, 3.4, 5.6, 7.8, 9.0, 2.3, 4.5, 6.7, 8.9, 1.1};
        double mean = calculateMean(data);
        double median = calculateMedian(data);
        double standardDeviation = calculateStandardDeviation(data, mean);
        
        System.out.println("Mean: " + mean);
        System.out.println("Median: " + median);
        System.out.println("Standard Deviation: " + standardDeviation);
        
        // Number theory operations
        System.out.println("GCD of 48 and 18: " + gcd(48, 18));
        System.out.println("LCM of 48 and 18: " + lcm(48, 18));
        System.out.println("Is 97 prime? " + isPrime(97));
        System.out.println("Prime factors of 315: " + getPrimeFactors(315));
    }

    public void testStringProcessing() {
        System.out.println("\n=== String Processing ===");
        
        String text = "The Quick Brown Fox Jumps Over The Lazy Dog";
        
        // String analysis
        Map<Character, Integer> charFrequency = getCharacterFrequency(text);
        System.out.println("Character frequency: " + charFrequency);
        
        String reversed = reverseWords(text);
        System.out.println("Reversed words: " + reversed);
        
        String encoded = caesarCipher(text, 3);
        String decoded = caesarCipher(encoded, -3);
        System.out.println("Caesar cipher (shift 3): " + encoded);
        System.out.println("Decoded: " + decoded);
        
        // Pattern matching
        String pattern = "(?i)\\b[a-z]{3}\\b"; // 3-letter words (case insensitive)
        List<String> threeLetterWords = findMatches(text, pattern);
        System.out.println("Three-letter words: " + threeLetterWords);
        
        // String transformation
        String camelCase = toCamelCase("hello_world_from_java");
        String snakeCase = toSnakeCase("HelloWorldFromJava");
        System.out.println("CamelCase: " + camelCase);
        System.out.println("SnakeCase: " + snakeCase);
        
        // Levenshtein distance
        int distance = levenshteinDistance("kitten", "sitting");
        System.out.println("Levenshtein distance between 'kitten' and 'sitting': " + distance);
    }

    public void testCollectionManipulation() {
        System.out.println("\n=== Collection Manipulation ===");
        
        // Multi-level nested collections
        Map<String, Map<String, List<Integer>>> nestedStructure = new HashMap<>();
        
        for (int i = 1; i <= 3; i++) {
            Map<String, List<Integer>> innerMap = new HashMap<>();
            for (int j = 1; j <= 3; j++) {
                List<Integer> values = new ArrayList<>();
                for (int k = 1; k <= 5; k++) {
                    values.add(i * j * k);
                }
                innerMap.put("key" + j, values);
            }
            nestedStructure.put("group" + i, innerMap);
        }
        
        // Flatten the nested structure
        List<Integer> flattened = flattenNestedStructure(nestedStructure);
        System.out.println("Flattened values: " + flattened);
        
        // Custom sorting with multiple criteria
        List<Person> people = Arrays.asList(
            new Person("Alice", 30, "Engineer"),
            new Person("Bob", 25, "Designer"),
            new Person("Charlie", 35, "Manager"),
            new Person("Diana", 25, "Engineer"),
            new Person("Eve", 30, "Designer")
        );
        
        // Sort by profession, then by age, then by name
        people.sort(Comparator
            .comparing(Person::getProfession)
            .thenComparing(Person::getAge)
            .thenComparing(Person::getName));
        
        System.out.println("Sorted people: " + people);
        
        // Set operations with custom objects
        Set<Person> engineers = people.stream()
            .filter(p -> "Engineer".equals(p.getProfession()))
            .collect(Collectors.toSet());
        
        Set<Person> youngPeople = people.stream()
            .filter(p -> p.getAge() < 30)
            .collect(Collectors.toSet());
        
        Set<Person> youngEngineers = new HashSet<>(engineers);
        youngEngineers.retainAll(youngPeople);
        
        System.out.println("Young engineers: " + youngEngineers);
        
        // Advanced stream operations
        Map<String, Double> avgAgeByProfession = people.stream()
            .collect(Collectors.groupingBy(
                Person::getProfession,
                Collectors.averagingDouble(Person::getAge)
            ));
        
        System.out.println("Average age by profession: " + avgAgeByProfession);
    }

    public void testFunctionalProgramming() {
        System.out.println("\n=== Functional Programming ===");
        
        // Higher-order functions
        Function<Integer, Integer> square = x -> x * x;
        Function<Integer, Integer> addTen = x -> x + 10;
        Function<Integer, Integer> composed = square.andThen(addTen);
        
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        List<Integer> transformed = numbers.stream()
            .map(composed)
            .collect(Collectors.toList());
        
        System.out.println("Transformed numbers (square then add 10): " + transformed);
        
        // Predicate composition
        Predicate<Integer> isEven = x -> x % 2 == 0;
        Predicate<Integer> isPositive = x -> x > 0;
        Predicate<Integer> isEvenAndPositive = isEven.and(isPositive);
        
        List<Integer> filtered = Arrays.asList(-4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
            .stream()
            .filter(isEvenAndPositive)
            .collect(Collectors.toList());
        
        System.out.println("Even and positive numbers: " + filtered);
        
        // Function currying simulation
        Function<Integer, Function<Integer, Integer>> curriedAdd = x -> y -> x + y;
        Function<Integer, Integer> addFive = curriedAdd.apply(5);
        
        List<Integer> addedFive = numbers.stream()
            .map(addFive)
            .collect(Collectors.toList());
        
        System.out.println("Numbers with 5 added: " + addedFive);
        
        // Lazy evaluation with suppliers
        Supplier<String> expensiveOperation = () -> {
            try {
                Thread.sleep(100); // Simulate expensive operation
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "Expensive result";
        };
        
        Optional<String> result = Optional.of("test")
            .filter(s -> s.length() > 3)
            .map(s -> expensiveOperation.get());
        
        System.out.println("Lazy evaluation result: " + result.orElse("No result"));
        
        // Memoization
        Function<Integer, BigInteger> memoizedFactorial = memoize(this::factorial);
        System.out.println("Factorial of 20: " + memoizedFactorial.apply(20));
        System.out.println("Factorial of 25: " + memoizedFactorial.apply(25));
    }

    public void testCryptographicOperations() {
        System.out.println("\n=== Cryptographic Operations ===");
        
        try {
            String plaintext = "This is a secret message";
            
            // Hash functions
            String md5Hash = generateHash(plaintext, "MD5");
            String sha256Hash = generateHash(plaintext, "SHA-256");
            String sha512Hash = generateHash(plaintext, "SHA-512");
            
            System.out.println("MD5: " + md5Hash);
            System.out.println("SHA-256: " + sha256Hash);
            System.out.println("SHA-512: " + sha512Hash);
            
            // Base64 encoding/decoding
            String encoded = Base64.getEncoder().encodeToString(plaintext.getBytes());
            String decoded = new String(Base64.getDecoder().decode(encoded));
            
            System.out.println("Base64 encoded: " + encoded);
            System.out.println("Base64 decoded: " + decoded);
            
            // Simple XOR cipher
            String key = "SECRET_KEY";
            String encrypted = xorEncrypt(plaintext, key);
            String decrypted = xorEncrypt(encrypted, key);
            
            System.out.println("XOR encrypted: " + encrypted);
            System.out.println("XOR decrypted: " + decrypted);
            
            // Random number generation
            SecureRandom secureRandom = new SecureRandom();
            byte[] randomBytes = new byte[16];
            secureRandom.nextBytes(randomBytes);
            
            String randomHex = bytesToHex(randomBytes);
            System.out.println("Random bytes (hex): " + randomHex);
            
        } catch (NoSuchAlgorithmException e) {
            System.err.println("Cryptographic algorithm not available: " + e.getMessage());
        }
    }

    public void testConcurrentOperations() {
        System.out.println("\n=== Concurrent Operations ===");
        
        // ExecutorService for parallel processing
        ExecutorService executor = Executors.newFixedThreadPool(4);
        
        List<Future<Integer>> futures = new ArrayList<>();
        
        // Submit multiple computational tasks
        for (int i = 1; i <= 10; i++) {
            final int taskId = i;
            Future<Integer> future = executor.submit(() -> {
                int result = expensiveComputation(taskId);
                System.out.println("Task " + taskId + " completed with result: " + result);
                return result;
            });
            futures.add(future);
        }
        
        // Collect results
        List<Integer> results = new ArrayList<>();
        for (Future<Integer> future : futures) {
            try {
                results.add(future.get());
            } catch (InterruptedException | ExecutionException e) {
                System.err.println("Task execution error: " + e.getMessage());
            }
        }
        
        executor.shutdown();
        
        System.out.println("All task results: " + results);
        
        // Producer-Consumer pattern
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);
        
        // Producer
        Thread producer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                try {
                    String item = "Item " + i;
                    queue.put(item);
                    System.out.println("Produced: " + item);
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
        
        // Consumer
        Thread consumer = new Thread(() -> {
            for (int i = 1; i <= 5; i++) {
                try {
                    String item = queue.take();
                    System.out.println("Consumed: " + item);
                    Thread.sleep(150);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
        
        producer.start();
        consumer.start();
        
        try {
            producer.join();
            consumer.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Atomic operations
        testAtomicOperations();
    }

    public void testDataValidation() {
        System.out.println("\n=== Data Validation ===");
        
        // Email validation
        String[] emails = {
            "valid@example.com",
            "invalid.email",
            "test@domain.co.uk",
            "user+tag@domain.com",
            "@invalid.com",
            "valid123@test-domain.org"
        };
        
        for (String email : emails) {
            boolean isValid = validateEmail(email);
            System.out.println("Email '" + email + "' is " + (isValid ? "valid" : "invalid"));
        }
        
        // Phone number validation
        String[] phoneNumbers = {
            "+1-234-567-8900",
            "(234) 567-8900",
            "234.567.8900",
            "2345678900",
            "123-45-6789",
            "+44 20 7946 0958"
        };
        
        for (String phone : phoneNumbers) {
            boolean isValid = validatePhoneNumber(phone);
            System.out.println("Phone '" + phone + "' is " + (isValid ? "valid" : "invalid"));
        }
        
        // Credit card number validation (Luhn algorithm)
        String[] creditCards = {
            "4532015112830366", // Valid Visa
            "5555555555554444", // Valid MasterCard
            "378282246310005",  // Valid American Express
            "1234567890123456", // Invalid
            "4532015112830367"  // Invalid (wrong check digit)
        };
        
        for (String card : creditCards) {
            boolean isValid = validateCreditCard(card);
            System.out.println("Credit card '" + card + "' is " + (isValid ? "valid" : "invalid"));
        }
        
        // Date validation and parsing
        String[] dates = {
            "2023-12-25",
            "2023-02-29", // Invalid (not a leap year)
            "2024-02-29", // Valid (leap year)
            "2023-13-01", // Invalid month
            "2023-12-32"  // Invalid day
        };
        
        for (String date : dates) {
            boolean isValid = validateDate(date);
            System.out.println("Date '" + date + "' is " + (isValid ? "valid" : "invalid"));
        }
    }

    public void testPerformanceMeasurement() {
        System.out.println("\n=== Performance Measurement ===");
        
        // Measure different sorting algorithms
        int[] data = generateRandomArray(10000);
        
        long startTime = System.nanoTime();
        int[] bubbleSorted = data.clone();
        bubbleSort(bubbleSorted);
        long bubbleTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int[] quickSorted = data.clone();
        quickSort(quickSorted, 0, quickSorted.length - 1);
        long quickTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int[] javaSorted = data.clone();
        Arrays.sort(javaSorted);
        long javaTime = System.nanoTime() - startTime;
        
        System.out.println("Bubble sort time: " + (bubbleTime / 1_000_000) + " ms");
        System.out.println("Quick sort time: " + (quickTime / 1_000_000) + " ms");
        System.out.println("Java sort time: " + (javaTime / 1_000_000) + " ms");
        
        // Memory usage estimation
        Runtime runtime = Runtime.getRuntime();
        long beforeMemory = runtime.totalMemory() - runtime.freeMemory();
        
        // Create large data structure
        List<String> largeList = new ArrayList<>();
        for (int i = 0; i < 100000; i++) {
            largeList.add("String " + i);
        }
        
        long afterMemory = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = afterMemory - beforeMemory;
        
        System.out.println("Memory used by large list: " + (memoryUsed / 1024) + " KB");
        
        // Clear the list to free memory
        largeList.clear();
        System.gc(); // Suggest garbage collection
    }

    // Helper classes and methods
    private static class ComplexNumber {
        private final double real;
        private final double imaginary;
        
        public ComplexNumber(double real, double imaginary) {
            this.real = real;
            this.imaginary = imaginary;
        }
        
        public ComplexNumber add(ComplexNumber other) {
            return new ComplexNumber(this.real + other.real, this.imaginary + other.imaginary);
        }
        
        public ComplexNumber multiply(ComplexNumber other) {
            double newReal = this.real * other.real - this.imaginary * other.imaginary;
            double newImaginary = this.real * other.imaginary + this.imaginary * other.real;
            return new ComplexNumber(newReal, newImaginary);
        }
        
        public double magnitude() {
            return Math.sqrt(real * real + imaginary * imaginary);
        }
        
        @Override
        public String toString() {
            return real + " + " + imaginary + "i";
        }
    }
    
    private static class Person {
        private final String name;
        private final int age;
        private final String profession;
        
        public Person(String name, int age, String profession) {
            this.name = name;
            this.age = age;
            this.profession = profession;
        }
        
        public String getName() { return name; }
        public int getAge() { return age; }
        public String getProfession() { return profession; }
        
        @Override
        public String toString() {
            return name + "(" + age + ", " + profession + ")";
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Person person = (Person) obj;
            return age == person.age && 
                   Objects.equals(name, person.name) && 
                   Objects.equals(profession, person.profession);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(name, age, profession);
        }
    }

    // Mathematical utility methods
    private double calculateMean(double[] data) {
        return Arrays.stream(data).average().orElse(0.0);
    }

    private double calculateMedian(double[] data) {
        double[] sorted = data.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }

    private double calculateStandardDeviation(double[] data, double mean) {
        double sum = 0.0;
        for (double value : data) {
            sum += Math.pow(value - mean, 2);
        }
        return Math.sqrt(sum / data.length);
    }

    private int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return Math.abs(a);
    }

    private int lcm(int a, int b) {
        return Math.abs(a * b) / gcd(a, b);
    }

    private boolean isPrime(int n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (int i = 3; i <= Math.sqrt(n); i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    private List<Integer> getPrimeFactors(int n) {
        List<Integer> factors = new ArrayList<>();
        for (int i = 2; i <= Math.sqrt(n); i++) {
            while (n % i == 0) {
                factors.add(i);
                n /= i;
            }
        }
        if (n > 1) factors.add(n);
        return factors;
    }

    private BigInteger factorial(int n) {
        BigInteger result = BigInteger.ONE;
        for (int i = 2; i <= n; i++) {
            result = result.multiply(BigInteger.valueOf(i));
        }
        return result;
    }

    // String processing methods
    private Map<Character, Integer> getCharacterFrequency(String text) {
        Map<Character, Integer> frequency = new HashMap<>();
        for (char c : text.toLowerCase().toCharArray()) {
            if (Character.isLetter(c)) {
                frequency.put(c, frequency.getOrDefault(c, 0) + 1);
            }
        }
        return frequency;
    }

    private String reverseWords(String text) {
        String[] words = text.split("\\s+");
        Collections.reverse(Arrays.asList(words));
        return String.join(" ", words);
    }

    private String caesarCipher(String text, int shift) {
        StringBuilder result = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (Character.isLetter(c)) {
                char base = Character.isUpperCase(c) ? 'A' : 'a';
                c = (char) (((c - base + shift + 26) % 26) + base);
            }
            result.append(c);
        }
        return result.toString();
    }

    private List<String> findMatches(String text, String pattern) {
        List<String> matches = new ArrayList<>();
        java.util.regex.Pattern compiledPattern = java.util.regex.Pattern.compile(pattern);
        java.util.regex.Matcher matcher = compiledPattern.matcher(text);
        while (matcher.find()) {
            matches.add(matcher.group());
        }
        return matches;
    }

    private String toCamelCase(String snakeCase) {
        String[] words = snakeCase.split("_");
        StringBuilder result = new StringBuilder(words[0].toLowerCase());
        for (int i = 1; i < words.length; i++) {
            result.append(words[i].substring(0, 1).toUpperCase())
                  .append(words[i].substring(1).toLowerCase());
        }
        return result.toString();
    }

    private String toSnakeCase(String camelCase) {
        return camelCase.replaceAll("([a-z])([A-Z])", "$1_$2").toLowerCase();
    }

    private int levenshteinDistance(String s1, String s2) {
        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        
        for (int i = 0; i <= s1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= s2.length(); j++) {
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= s1.length(); i++) {
            for (int j = 1; j <= s2.length(); j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        
        return dp[s1.length()][s2.length()];
    }

    // Collection utility methods
    private List<Integer> flattenNestedStructure(Map<String, Map<String, List<Integer>>> nested) {
        return nested.values().stream()
            .flatMap(innerMap -> innerMap.values().stream())
            .flatMap(List::stream)
            .collect(Collectors.toList());
    }

    // Functional programming utilities
    private <T, R> Function<T, R> memoize(Function<T, R> function) {
        Map<T, R> cache = new ConcurrentHashMap<>();
        return input -> cache.computeIfAbsent(input, function);
    }

    // Cryptographic utilities
    private String generateHash(String input, String algorithm) throws NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance(algorithm);
        byte[] hash = digest.digest(input.getBytes());
        return bytesToHex(hash);
    }

    private String xorEncrypt(String text, String key) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            result.append((char) (text.charAt(i) ^ key.charAt(i % key.length())));
        }
        return result.toString();
    }

    private String bytesToHex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte b : bytes) {
            result.append(String.format("%02x", b));
        }
        return result.toString();
    }

    // Concurrent utilities
    private int expensiveComputation(int input) {
        try {
            Thread.sleep(100 + (int)(Math.random() * 200)); // Simulate work
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return input * input + input;
    }

    private void testAtomicOperations() {
        java.util.concurrent.atomic.AtomicInteger atomicCounter = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicReference<String> atomicString = new java.util.concurrent.atomic.AtomicReference<>("initial");
        
        List<Thread> threads = new ArrayList<>();
        
        for (int i = 0; i < 5; i++) {
            Thread thread = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    atomicCounter.incrementAndGet();
                }
                atomicString.updateAndGet(s -> s + ".");
            });
            threads.add(thread);
            thread.start();
        }
        
        threads.forEach(t -> {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        System.out.println("Atomic counter final value: " + atomicCounter.get());
        System.out.println("Atomic string final value: " + atomicString.get());
    }

    // Validation methods
    private boolean validateEmail(String email) {
        String emailRegex = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
        return email.matches(emailRegex);
    }

    private boolean validatePhoneNumber(String phone) {
        String phoneRegex = "^[+]?[1-9]\\d{0,3}[-.\\s]?\\(?\\d{1,3}\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}$";
        return phone.replaceAll("[^+\\d]", "").matches("^[+]?\\d{10,15}$");
    }

    private boolean validateCreditCard(String cardNumber) {
        cardNumber = cardNumber.replaceAll("\\s+", "");
        if (!cardNumber.matches("\\d+")) return false;
        
        // Luhn algorithm
        int sum = 0;
        boolean alternate = false;
        for (int i = cardNumber.length() - 1; i >= 0; i--) {
            int digit = Character.getNumericValue(cardNumber.charAt(i));
            if (alternate) {
                digit *= 2;
                if (digit > 9) digit -= 9;
            }
            sum += digit;
            alternate = !alternate;
        }
        return sum % 10 == 0;
    }

    private boolean validateDate(String date) {
        try {
            java.time.LocalDate.parse(date);
            return true;
        } catch (java.time.format.DateTimeParseException e) {
            return false;
        }
    }

    // Performance testing utilities
    private int[] generateRandomArray(int size) {
        Random random = new Random();
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextInt(10000);
        }
        return array;
    }

    private void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    private void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pivot = partition(arr, low, high);
            quickSort(arr, low, pivot - 1);
            quickSort(arr, pivot + 1, high);
        }
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    // Getters and setters
    public String getIdentifier() { return identifier; }
    public void setIdentifier(String identifier) { this.identifier = identifier; }
    public BigDecimal getBalance() { return balance; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
    public Set<String> getTags() { return tags; }
    public void setTags(Set<String> tags) { this.tags = tags; }
    public Queue<String> getOperations() { return operations; }
    public void setOperations(Queue<String> operations) { this.operations = operations; }

    @Override
    public String toString() {
        return "TestFile2{identifier='" + identifier + "', balance=" + balance + ", tags=" + tags + "}";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        TestFile2 testFile2 = (TestFile2) obj;
        return Objects.equals(identifier, testFile2.identifier) && 
               Objects.equals(balance, testFile2.balance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(identifier, balance);
    }
}
