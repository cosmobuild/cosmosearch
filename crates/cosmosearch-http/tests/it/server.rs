use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use serde_json::Value;

pub(crate) enum Reply {
    Answer(u16, String),
    Asking(u16, String, String),
    Declaring(u64),
    Unsized(usize),
    Hang,
}

pub(crate) struct Captured {
    pub(crate) head: String,
    pub(crate) body: Value,
}

pub(crate) struct Server {
    pub(crate) base: String,
    pub(crate) requests: Receiver<Captured>,
}

impl Server {
    pub(crate) fn next(&self) -> Captured {
        self.requests.recv().unwrap()
    }
}

pub(crate) fn serve(replies: Vec<Reply>) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let (sender, requests) = mpsc::channel();
    thread::spawn(move || {
        for reply in replies {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            sender.send(read_request(&mut reader)).unwrap();
            let mut stream = reader.into_inner();
            match reply {
                Reply::Answer(status, body) => answer(&mut stream, status, "", &body),
                Reply::Asking(status, header, body) => {
                    answer(&mut stream, status, &format!("{header}\r\n"), &body);
                }
                Reply::Declaring(length) => {
                    let head = format!(
                        "HTTP/1.1 200 Canned\r\ncontent-length: {length}\r\nconnection: close\r\n\r\n"
                    );
                    stream.write_all(head.as_bytes()).unwrap_or_default();
                }
                Reply::Unsized(length) => {
                    let head = "HTTP/1.1 200 Canned\r\nconnection: close\r\n\r\n";
                    stream.write_all(head.as_bytes()).unwrap_or_default();
                    stream.write_all(&vec![b' '; length]).unwrap_or_default();
                }
                Reply::Hang => {
                    io::copy(&mut stream, &mut io::sink()).unwrap_or_default();
                }
            }
        }
    });
    Server { base, requests }
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Captured {
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        head.push_str(&line.to_ascii_lowercase());
    }
    let length = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length: "))
        .map_or(0, |value| value.trim().parse().unwrap());
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).unwrap();
    Captured {
        head,
        body: serde_json::from_slice(&body).unwrap(),
    }
}

fn answer(stream: &mut TcpStream, status: u16, headers: &str, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} Canned\r\ncontent-type: application/json\r\n{headers}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).unwrap();
}
