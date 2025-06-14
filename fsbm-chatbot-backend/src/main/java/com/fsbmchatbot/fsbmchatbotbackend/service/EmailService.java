package com.fsbmchatbot.fsbmchatbotbackend.service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {

    private final JavaMailSender mailSender;

    @Autowired
    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

   public boolean sendSimpleEmail(String to, String subject, String body) {
    try {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom("fsbmscholarassistant@gmail.com");
        message.setTo(to);
        message.setSubject(subject);
        message.setText("Bonjour,\n\nVotre code de vérification est : " + body + "\n\nCordialement.");
        mailSender.send(message);
        return true;
    } catch (Exception e) {
        e.printStackTrace();
        return false;
    }
}
}
