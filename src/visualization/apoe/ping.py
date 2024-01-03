import requests
import time

def ping_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Success! {url} is reachable.")
        else:
            print(f"Failed! {url} returned status code: {response.status_code}")
    except requests.ConnectionError:
        print(f"Failed! {url} is unreachable.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    url = "https://oanda4.com"  # Replace with the URL you want to ping
    interval = 0  # Time interval between each ping in seconds

    while True:
        ping_website(url)
        time.sleep(interval)

if __name__ == "__main__":
    main()
