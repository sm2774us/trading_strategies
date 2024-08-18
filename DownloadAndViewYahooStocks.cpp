// DownloadAndViewYahooStocks.cpp
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <matplot/matplot.h>

using json = nlohmann::json;

// Callback function to handle the data received by curl
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

json download_stock_data(const std::string& symbol, const std::string& start_date) {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl = curl_easy_init();
    if (curl) {
        std::ostringstream url;
        url << "https://query1.finance.yahoo.com/v7/finance/download/"
            << symbol
            << "?period1=1577836800&period2=9999999999&interval=1d&events=history";

        curl_easy_setopt(curl, CURLOPT_URL, url.str().c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
    }

    // Parse CSV data to JSON
    std::istringstream dataStream(readBuffer);
    std::string line;
    json data = json::array();

    // Skip the header
    std::getline(dataStream, line);

    while (std::getline(dataStream, line)) {
        std::istringstream lineStream(line);
        std::string date, open, high, low, close, adj_close, volume;
        std::getline(lineStream, date, ',');
        std::getline(lineStream, open, ',');
        std::getline(lineStream, high, ',');
        std::getline(lineStream, low, ',');
        std::getline(lineStream, close, ',');
        std::getline(lineStream, adj_close, ',');
        std::getline(lineStream, volume, ',');

        data.push_back({
            {"Date", date},
            {"Open", std::stod(open)},
            {"High", std::stod(high)},
            {"Low", std::stod(low)},
            {"Close", std::stod(close)},
            {"Adj Close", std::stod(adj_close)},
            {"Volume", std::stoll(volume)}
        });
    }

    return data;
}

Eigen::VectorXd calculate_sma(const Eigen::VectorXd& close_prices, int window) {
    Eigen::VectorXd sma = Eigen::VectorXd::Zero(close_prices.size());
    for (int i = window - 1; i < close_prices.size(); ++i) {
        sma(i) = close_prices.segment(i - window + 1, window).mean();
    }
    return sma;
}

Eigen::VectorXd calculate_stddev(const Eigen::VectorXd& close_prices, const Eigen::VectorXd& sma, int window) {
    Eigen::VectorXd stddev = Eigen::VectorXd::Zero(close_prices.size());
    for (int i = window - 1; i < close_prices.size(); ++i) {
        stddev(i) = std::sqrt((close_prices.segment(i - window + 1, window).array() - sma(i)).square().mean());
    }
    return stddev;
}

struct Indicators {
    Eigen::VectorXd sma;
    Eigen::VectorXd upper_band;
    Eigen::VectorXd lower_band;
};

Indicators calculate_bollinger_bands(const Eigen::VectorXd& close_prices, int window) {
    Eigen::VectorXd sma = calculate_sma(close_prices, window);
    Eigen::VectorXd stddev = calculate_stddev(close_prices, sma, window);
    Indicators ind;
    ind.sma = sma;
    ind.upper_band = sma + 2 * stddev;
    ind.lower_band = sma - 2 * stddev;
    return ind;
}

void plot_stock_data(const json& data, const Indicators& indicators) {
    using namespace matplot;

    // Prepare the data for plotting
    std::vector<double> dates;
    std::vector<double> opens, highs, lows, closes;
    std::vector<double> sma(indicators.sma.data(), indicators.sma.data() + indicators.sma.size());
    std::vector<double> upper_band(indicators.upper_band.data(), indicators.upper_band.data() + indicators.upper_band.size());
    std::vector<double> lower_band(indicators.lower_band.data(), indicators.lower_band.data() + indicators.lower_band.size());

    for (const auto& entry : data) {
        dates.push_back(date::year_month_day{date::year(entry["Date"].get<std::string>().substr(0, 4)), 
                                             date::month(entry["Date"].get<std::string>().substr(5, 2)), 
                                             date::day(entry["Date"].get<std::string>().substr(8, 2))}.to_time_point().time_since_epoch().count());
        opens.push_back(entry["Open"]);
        highs.push_back(entry["High"]);
        lows.push_back(entry["Low"]);
        closes.push_back(entry["Close"]);
    }

    // Plot candlesticks
    auto ax = subplot(1, 1, 0);
    ax->title("AAPL Stock Price with Bollinger Bands");

    auto up = ax->hold(true);
    ax->bar(dates, closes, 0.6, opens, closes, "", "g");
    ax->bar(dates, highs, 0.05, closes, highs, "", "g");
    ax->bar(dates, lows, 0.05, opens, lows, "", "g");
    ax->bar(dates, closes, 0.6, opens, closes, "", "r");
    ax->bar(dates, highs, 0.05, opens, highs, "", "r");
    ax->bar(dates, lows, 0.05, closes, lows, "", "r");

    // Plot SMA, Upper, and Lower bands
    ax->plot(dates, sma, "b")->line_width(2).display_name("SMA");
    ax->plot(dates, upper_band, "--", "color", "gray")->line_width(2).display_name("Upper");
    ax->plot(dates, lower_band, "--", "color", "gray")->line_width(2).display_name("Lower");

    ax->legend();
    ax->xlabel("Date");
    ax->ylabel("Price");
    ax->grid(true);

    matplot::show();
}

int main() {
    // Download stock data
    json stock_data = download_stock_data("AAPL", "2020-01-01");

    // Convert close prices to Eigen vector
    Eigen::VectorXd close_prices(stock_data.size());
    for (size_t i = 0; i < stock_data.size(); ++i) {
        close_prices(i) = stock_data[i]["Close"];
    }

    // Calculate indicators
    Indicators indicators = calculate_bollinger_bands(close_prices, 20);

    // Plot data
    plot_stock_data(stock_data, indicators);

    return 0;
}